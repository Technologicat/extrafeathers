# -*- coding: utf-8; -*-
"""Miscellaneous utilities."""

import matplotlib
import matplotlib.pyplot as plt

# Matplotlib (3.3.3) has a habit of popping the figure window to top when it is updated using show() or pause(),
# which effectively prevents using the machine for anything else while a simulation is in progress.
#
# To fix this, the suggestion to use the Qt5Agg backend here:
#   https://stackoverflow.com/questions/61397176/how-to-keep-matplotlib-from-stealing-focus
#
# didn't help on my system (Linux Mint 20.1). And it is somewhat nontrivial to use a `FuncAnimation` here.
# So we'll use this custom pause function hack instead, courtesy of StackOverflow user @ImportanceOfBeingErnest:
#   https://stackoverflow.com/a/45734500
#
def mypause(interval: float) -> None:
    """Redraw the current figure without stealing focus.

    Works after `plt.show()` has been called at least once.
    """
    backend = plt.rcParams['backend']
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw_idle()
            canvas.start_event_loop(interval)
