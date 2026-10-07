"""The runnable application - two front-ends over one pipeline.

    cli        `python -m invoice_extractor` - PDFs in, JSON out
    frontend   `python -m invoice_extractor.app.frontend` - the same run behind a window

Both call `core.pipeline` and nothing else, so the GUI cannot drift from the command line.
Kept as submodules with no eager re-export, so importing `app` does not pull in Tkinter on a
machine that has no display.
"""
