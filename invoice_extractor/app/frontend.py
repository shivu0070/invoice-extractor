"""Tkinter front-end: the same run as the command line, behind one window.

    python -m invoice_extractor.app.frontend

One window, top to bottom - the form, the button bar, a progress bar, a result strip that
appears only once a run has finished, and the pipeline log. Nothing is hidden behind a tab and
nothing switches view on its own, so the log stays readable while a run is going.

It calls `core.pipeline.process_pdf` and nothing else, which is the point: the window cannot
drift away from what `app.cli` does, because there is only one code path underneath.

**Stop** costs the pipeline nothing. `process_pdf` already calls back on every page it reads
and every page it extracts; raising out of one of those callbacks unwinds the run at the next
page boundary. So cancellation is implemented here, in the front-end, and `core` has no idea
the GUI exists. The in-flight page still finishes - a model call cannot be interrupted midway -
so Stop takes effect within one page, not instantly.

Tkinter is single-threaded and unsafe to touch from anywhere but the main loop, so the run
happens on a worker thread that only ever puts messages on a `queue.Queue`, which the main loop
drains on a timer.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from ..core.config import LLM_MODEL, OCR_MODEL, OLLAMA_URL
from ..core.pipeline import PageResult, process_pdf
from ..preprocessing import page_images

PDF_TYPES = [("PDF files", "*.pdf"), ("All files", "*.*")]

#: Log colours. Dark because the log is the part you stare at while a page is being read.
LOG_BG = "#12141a"
LOG_FG = "#d8dee9"
LOG_STAGE = "#56b6c2"
LOG_BAD = "#e06c75"
LOG_OK = "#98c379"
LOG_DIM = "#7d8799"


class _Cancelled(Exception):
    """Raised inside a pipeline callback to unwind a run the user stopped."""


def _open_folder(path: Path) -> None:
    """Show a folder in the system file manager, on whichever platform this is."""
    if sys.platform == "win32":
        os.startfile(str(path))                                   # noqa: S606
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])                     # noqa: S603,S607
    else:
        subprocess.Popen(["xdg-open", str(path)])                 # noqa: S603,S607


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("invoice extractor")
        self.geometry("980x680")
        self.minsize(760, 520)

        self._q: queue.Queue = queue.Queue()
        self._cancel = threading.Event()
        self._worker: threading.Thread | None = None
        self._pdfs: list[Path] = []
        self._last_results: list[tuple[Path, PageResult]] = []

        self._build()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------------------------------------------------------------- layout

    def _build(self) -> None:
        pad = {"padx": 10, "pady": 4}

        form = ttk.LabelFrame(self, text="input")
        form.pack(fill="x", **pad)
        form.columnconfigure(1, weight=1)

        self.v_input = tk.StringVar(value="")
        self.v_out = tk.StringVar(value=str(Path.cwd() / "out"))
        self.v_host = tk.StringVar(value=OLLAMA_URL)
        self.v_ocr = tk.StringVar(value=OCR_MODEL)
        self.v_llm = tk.StringVar(value=LLM_MODEL)
        self.v_md = tk.BooleanVar(value=False)

        ttk.Label(form, text="PDFs").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(form, textvariable=self.v_input).grid(row=0, column=1, sticky="ew", pady=6)
        picks = ttk.Frame(form)
        picks.grid(row=0, column=2, sticky="e", padx=8)
        ttk.Button(picks, text="Files...", width=9,
                   command=self._pick_files).pack(side="left", padx=2)
        ttk.Button(picks, text="Folder...", width=9,
                   command=self._pick_folder).pack(side="left", padx=2)

        ttk.Label(form, text="output").grid(row=1, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(form, textvariable=self.v_out).grid(row=1, column=1, sticky="ew", pady=6)
        ttk.Button(form, text="Choose...", width=20,
                   command=self._pick_out).grid(row=1, column=2, sticky="e", padx=8)

        adv = ttk.LabelFrame(self, text="models")
        adv.pack(fill="x", **pad)
        adv.columnconfigure(1, weight=1)
        adv.columnconfigure(3, weight=1)
        ttk.Label(adv, text="host").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(adv, textvariable=self.v_host).grid(row=0, column=1, columnspan=3,
                                                      sticky="ew", padx=(0, 8), pady=6)
        ttk.Label(adv, text="ocr").grid(row=1, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(adv, textvariable=self.v_ocr).grid(row=1, column=1, sticky="ew", pady=6)
        ttk.Label(adv, text="llm").grid(row=1, column=2, sticky="w", padx=8, pady=6)
        ttk.Entry(adv, textvariable=self.v_llm).grid(row=1, column=3, sticky="ew",
                                                     padx=(0, 8), pady=6)
        ttk.Checkbutton(adv, text="also write the vision model's markdown "
                                  "(the first thing to look at when a page comes out wrong)",
                        variable=self.v_md).grid(row=2, column=0, columnspan=4,
                                                 sticky="w", padx=8, pady=(0, 6))

        bar = ttk.Frame(self)
        bar.pack(fill="x", **pad)
        self.b_scan = ttk.Button(bar, text="Scan", width=14, command=self._start)
        self.b_scan.pack(side="left")
        self.b_stop = ttk.Button(bar, text="Stop", width=10,
                                 command=self._stop, state="disabled")
        self.b_stop.pack(side="left", padx=6)
        ttk.Button(bar, text="Open output", width=14,
                   command=self._open_out).pack(side="left")
        self.l_status = ttk.Label(bar, text="idle", foreground=LOG_DIM)
        self.l_status.pack(side="right")

        self.progress = ttk.Progressbar(self, mode="determinate")
        self.progress.pack(fill="x", padx=10, pady=(0, 4))

        # The result strip only means something once a run has ended, so it starts hidden.
        self.strip = ttk.Frame(self)
        self.chips: dict[str, ttk.Label] = {}
        for name in ("failed", "pages", "rows"):
            lab = ttk.Label(self.strip, text=f"{name} -", anchor="center",
                            relief="groove", padding=(14, 6))
            lab.pack(side="left", padx=(0, 8))
            self.chips[name] = lab
        ttk.Button(self.strip, text="Inspect pages",
                   command=self._open_viewer).pack(side="right")

        log_wrap = ttk.Frame(self)
        log_wrap.pack(fill="both", expand=True, padx=10, pady=(4, 10))
        self.log = tk.Text(log_wrap, bg=LOG_BG, fg=LOG_FG, insertbackground=LOG_FG,
                           font=("Consolas", 9), wrap="none", relief="flat",
                           padx=10, pady=8, state="disabled")
        sb = ttk.Scrollbar(log_wrap, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.log.pack(side="left", fill="both", expand=True)
        self.log.tag_configure("stage", foreground=LOG_STAGE)
        self.log.tag_configure("bad", foreground=LOG_BAD)
        self.log.tag_configure("ok", foreground=LOG_OK)
        self.log.tag_configure("dim", foreground=LOG_DIM)

    # ---------------------------------------------------------------- pickers

    def _pick_files(self) -> None:
        got = filedialog.askopenfilenames(title="choose PDFs", filetypes=PDF_TYPES)
        if got:
            self._pdfs = [Path(p) for p in got]
            self.v_input.set("; ".join(p.name for p in self._pdfs))

    def _pick_folder(self) -> None:
        got = filedialog.askdirectory(title="choose a folder of PDFs")
        if got:
            self._pdfs = sorted(Path(got).glob("*.pdf"))
            self.v_input.set(f"{got}  ({len(self._pdfs)} pdf)")

    def _pick_out(self) -> None:
        got = filedialog.askdirectory(title="where the JSON goes")
        if got:
            self.v_out.set(got)

    def _open_out(self) -> None:
        out = Path(self.v_out.get())
        if not out.is_dir():
            messagebox.showinfo("nothing yet", f"{out} does not exist - run a scan first.")
            return
        _open_folder(out)

    def _open_viewer(self) -> None:
        if not self._last_results:
            messagebox.showinfo("nothing to inspect", "Run a scan first.")
            return
        PageViewer(self, self._last_results)

    # ---------------------------------------------------------------- logging

    def _write(self, text: str, tag: str = "") -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n", tag or ())
        self.log.see("end")
        self.log.configure(state="disabled")

    # ---------------------------------------------------------------- the run

    def _start(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        pdfs = [p for p in self._pdfs if p.is_file()]
        if not pdfs:
            messagebox.showwarning("no PDFs", "Pick one or more PDF files, or a folder.")
            return

        out = Path(self.v_out.get())
        try:
            out.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            messagebox.showerror("output", f"cannot create {out}\n{e}")
            return

        self._cancel.clear()
        self._last_results = []
        self.strip.pack_forget()
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self.b_scan.configure(state="disabled")
        self.b_stop.configure(state="normal")
        self.l_status.configure(text="running", foreground=LOG_STAGE)
        self.progress.configure(value=0, maximum=len(pdfs))

        job = {
            "pdfs": pdfs, "out": out, "host": self.v_host.get().strip(),
            "ocr": self.v_ocr.get().strip(), "llm": self.v_llm.get().strip(),
            "keep_md": bool(self.v_md.get()),
        }
        self._worker = threading.Thread(target=self._run, args=(job,), daemon=True)
        self._worker.start()
        self.after(80, self._drain)

    def _stop(self) -> None:
        self._cancel.set()
        self.b_stop.configure(state="disabled")
        self.l_status.configure(text="stopping after this page", foreground=LOG_BAD)

    def _run(self, job: dict) -> None:
        """Worker thread. Touches no widget - everything goes through the queue."""
        put = self._q.put
        put(("log", f"ocr : {job['ocr']}", "dim"))
        put(("log", f"llm : {job['llm']}", "dim"))
        put(("log", f"host: {job['host']}", "dim"))

        n_pages = n_rows = n_err = 0
        t0 = time.time()
        cancelled = False

        for i, pdf in enumerate(job["pdfs"], 1):
            if self._cancel.is_set():
                cancelled = True
                break
            put(("log", f"== {pdf.name} ==", "stage"))

            def on_ocr(key: str, md: str) -> None:
                if self._cancel.is_set():
                    raise _Cancelled
                put(("log", f"   read {key[:52]:52s}  {len(md):5d} chars", ""))

            def on_page(res) -> None:
                nonlocal n_pages, n_rows, n_err
                if self._cancel.is_set():
                    raise _Cancelled
                n_pages += 1
                if res.error:
                    n_err += 1
                    put(("log", f"   FAIL {res.key[:52]:52s}  {res.error}", "bad"))
                else:
                    n_rows += len(res.rows)
                    put(("log", f"        {res.key[:52]:52s}  "
                                f"{len(res.rows)} row(s) x {len(res.columns)} col(s)", "ok"))

            try:
                results = process_pdf(pdf, host=job["host"], ocr_model=job["ocr"],
                                      llm_model=job["llm"], on_page=on_page, on_ocr=on_ocr)
            except _Cancelled:
                cancelled = True
                put(("log", "   cancelled", "bad"))
                break
            except Exception as e:                                # noqa: BLE001
                n_err += 1
                put(("log", f"   FAIL {pdf.name}: {e.__class__.__name__}: {e}", "bad"))
                put(("step", i))
                continue

            for res in results:
                safe = res.key.replace("#", "_")
                (job["out"] / f"{safe}.json").write_text(
                    json.dumps(res.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
                if job["keep_md"]:
                    (job["out"] / f"{safe}.md").write_text(res.markdown, encoding="utf-8")
                self._last_results.append((pdf, res))
            put(("step", i))

        put(("done", {"pages": n_pages, "rows": n_rows, "failed": n_err,
                      "secs": time.time() - t0, "cancelled": cancelled}))

    def _drain(self) -> None:
        """Main loop. Pulls whatever the worker has said and paints it."""
        try:
            while True:
                kind, *rest = self._q.get_nowait()
                if kind == "log":
                    self._write(rest[0], rest[1])
                elif kind == "step":
                    self.progress.configure(value=rest[0])
                elif kind == "done":
                    self._finish(rest[0])
                    return
        except queue.Empty:
            pass
        self.after(80, self._drain)

    def _finish(self, s: dict) -> None:
        self.b_scan.configure(state="normal")
        self.b_stop.configure(state="disabled")
        per = s["secs"] / max(s["pages"], 1)
        tail = "cancelled" if s["cancelled"] else "done"
        self._write("", "")
        self._write(f"{tail}: {s['pages']} page(s) in {s['secs']:.1f}s ({per:.1f}s/page)",
                    "bad" if s["cancelled"] else "stage")
        self.l_status.configure(text=tail,
                                foreground=LOG_BAD if s["cancelled"] else LOG_OK)
        for name in ("failed", "pages", "rows"):
            self.chips[name].configure(text=f"{name}  {s[name]}")
        self.strip.pack(fill="x", padx=10, pady=(0, 6), before=self.log.master)

    def _on_close(self) -> None:
        if self._worker and self._worker.is_alive():
            if not messagebox.askokcancel("quit", "A scan is running. Stop it and quit?"):
                return
            self._cancel.set()
        self.destroy()


class PageViewer(tk.Toplevel):
    """The source page beside what was extracted from it - a plain viewer, no verdicts yet.

    Re-renders each page from its PDF on demand (`page_images` is cheap - no OCR call) rather
    than the pipeline saving images to disk, so a scan's only side effect stays the JSON it
    already wrote. Rendering the SAME function `process_pdf` uses means what you see here is
    exactly what the OCR model saw, upright and deskewed.
    """

    ZOOM_STEP = 1.15
    ZOOM_MIN, ZOOM_MAX = 0.15, 4.0

    def __init__(self, master: tk.Misc, items: list[tuple[Path, PageResult]]) -> None:
        super().__init__(master)
        self.title("Inspect pages")
        self.geometry("1200x760")
        self.minsize(900, 560)

        self._items = items
        self._img_cache: dict[Path, dict[str, Image.Image]] = {}
        self._base_img: Image.Image | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._zoom = 1.0

        self._build()
        self._populate_list()

    def _build(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        panes = ttk.Panedwindow(self, orient="horizontal")
        panes.grid(row=0, column=0, sticky="nsew")

        left = ttk.Frame(panes)
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        self.list = ttk.Treeview(left, columns=("info",), show="tree headings",
                                 selectmode="browse")
        self.list.heading("#0", text="page")
        self.list.heading("info", text="")
        self.list.column("#0", width=260, stretch=True)
        self.list.column("info", width=80, anchor="center", stretch=False)
        self.list.grid(row=0, column=0, sticky="nsew")
        lsb = ttk.Scrollbar(left, orient="vertical", command=self.list.yview)
        lsb.grid(row=0, column=1, sticky="ns")
        self.list.configure(yscrollcommand=lsb.set)
        self.list.bind("<<TreeviewSelect>>", self._on_select)
        panes.add(left, weight=1)

        right = ttk.Panedwindow(panes, orient="vertical")
        panes.add(right, weight=3)

        img_frame = ttk.Frame(right)
        img_frame.columnconfigure(0, weight=1)
        img_frame.rowconfigure(1, weight=1)
        zoom_bar = ttk.Frame(img_frame)
        zoom_bar.grid(row=0, column=0, sticky="ew", pady=(0, 2))
        ttk.Button(zoom_bar, text="-", width=3,
                   command=lambda: self._zoom_by(1 / self.ZOOM_STEP)).pack(side="left")
        ttk.Button(zoom_bar, text="+", width=3,
                   command=lambda: self._zoom_by(self.ZOOM_STEP)).pack(side="left", padx=(4, 0))
        ttk.Button(zoom_bar, text="fit", width=5,
                   command=self._zoom_fit).pack(side="left", padx=(8, 0))
        ttk.Button(zoom_bar, text="100%", width=6,
                   command=lambda: self._set_zoom(1.0)).pack(side="left", padx=(4, 0))
        self.l_zoom = ttk.Label(zoom_bar, text="")
        self.l_zoom.pack(side="right")

        canvas_wrap = ttk.Frame(img_frame)
        canvas_wrap.grid(row=1, column=0, sticky="nsew")
        canvas_wrap.rowconfigure(0, weight=1)
        canvas_wrap.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(canvas_wrap, background="#2b2b2b", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vsb = ttk.Scrollbar(canvas_wrap, orient="vertical", command=self.canvas.yview)
        vsb.grid(row=0, column=1, sticky="ns")
        hsb = ttk.Scrollbar(canvas_wrap, orient="horizontal", command=self.canvas.xview)
        hsb.grid(row=1, column=0, sticky="ew")
        self.canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.canvas.bind("<MouseWheel>", self._on_wheel)                      # Windows / macOS
        self.canvas.bind("<Button-4>", lambda e: self._zoom_by(self.ZOOM_STEP))      # X11 up
        self.canvas.bind("<Button-5>", lambda e: self._zoom_by(1 / self.ZOOM_STEP))  # X11 down
        right.add(img_frame, weight=3)

        data_frame = ttk.Frame(right, padding=(0, 6, 0, 0))
        data_frame.columnconfigure(0, weight=1)
        data_frame.rowconfigure(1, weight=1)
        self.l_envelope = ttk.Label(data_frame, text="", justify="left", wraplength=760)
        self.l_envelope.grid(row=0, column=0, sticky="w", pady=(0, 4))
        self.table = ttk.Treeview(data_frame, show="headings", selectmode="none")
        self.table.grid(row=1, column=0, sticky="nsew")
        tsb = ttk.Scrollbar(data_frame, orient="vertical", command=self.table.yview)
        tsb.grid(row=1, column=1, sticky="ns")
        self.table.configure(yscrollcommand=tsb.set)
        right.add(data_frame, weight=2)

    # ---------------------------------------------------------------- list / selection

    def _populate_list(self) -> None:
        for _pdf, res in self._items:
            info = "ERROR" if res.error else f"{len(res.rows)}x{len(res.columns)}"
            self.list.insert("", "end", iid=res.key, text=res.key, values=(info,))
        if self._items:
            self.list.selection_set(self._items[0][1].key)

    def _on_select(self, _evt=None) -> None:
        sel = self.list.selection()
        if not sel:
            return
        for pdf, res in self._items:
            if res.key == sel[0]:
                self._show(pdf, res)
                return

    def _show(self, pdf: Path, res: PageResult) -> None:
        if pdf not in self._img_cache:
            try:
                self._img_cache[pdf] = dict(page_images(pdf))
            except Exception:                                        # noqa: BLE001
                self._img_cache[pdf] = {}
        self._base_img = self._img_cache[pdf].get(res.key)
        self._zoom_fit()
        self._fill_table(res)
        self._fill_envelope(res)

    def _fill_table(self, res: PageResult) -> None:
        cols = list(res.columns)
        self.table.configure(columns=cols)
        for c in cols:
            self.table.heading(c, text=c)
            self.table.column(c, width=max(70, min(260, 9 * len(c) + 30)), stretch=True)
        self.table.delete(*self.table.get_children())
        for row in res.rows:
            self.table.insert("", "end", values=[row.get(c, "") for c in cols])

    def _fill_envelope(self, res: PageResult) -> None:
        parts = [f"{k}: {v}" for k, v in (res.envelope or {}).items() if v]
        text = res.key
        if res.error:
            text += f"   [ERROR: {res.error}]"
        if parts:
            text += "\n" + "    ".join(parts)
        self.l_envelope.configure(text=text)

    # ---------------------------------------------------------------- zoom

    def _on_wheel(self, event) -> None:
        self._zoom_by(self.ZOOM_STEP if event.delta > 0 else 1 / self.ZOOM_STEP)

    def _zoom_by(self, factor: float) -> None:
        self._set_zoom(self._zoom * factor)

    def _zoom_fit(self) -> None:
        self.update_idletasks()
        if self._base_img is None:
            self._set_zoom(1.0)
            return
        cw = max(self.canvas.winfo_width(), 200)
        self._set_zoom(min(1.0, cw / self._base_img.width))

    def _set_zoom(self, z: float) -> None:
        self._zoom = max(self.ZOOM_MIN, min(self.ZOOM_MAX, z))
        self._render()

    def _render(self) -> None:
        self.canvas.delete("all")
        if self._base_img is None:
            self.canvas.create_text(12, 12, anchor="nw", fill="#ccc", text="(no page image)")
            self.l_zoom.configure(text="")
            return
        w = max(1, round(self._base_img.width * self._zoom))
        h = max(1, round(self._base_img.height * self._zoom))
        img = self._base_img.resize((w, h), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
        self.canvas.configure(scrollregion=(0, 0, w, h))
        self.l_zoom.configure(text=f"{self._zoom * 100:.0f}%")


def main(argv: list[str] | None = None) -> int:
    App().mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
