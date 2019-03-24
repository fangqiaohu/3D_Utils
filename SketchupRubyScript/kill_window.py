"""kill a windows system window, known window's name or corner point"""

from win32.lib import win32con
from win32gui import *
import time, sys


def kill_window(hwnd, mouse):
    if IsWindow(hwnd) and IsWindowEnabled(hwnd) and IsWindowVisible(hwnd):
        title = GetWindowText(hwnd)
        rect = GetWindowRect(hwnd)
        height, width = (rect[3]-rect[1], rect[2]-rect[0])
        # if titie == 'kmzimporter':
        if title == 'kmzimporter' and width < 600 or title == 'SketchUp' and width < 600:
            # close
            PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
            print("'%s' has been killed with width=%d." % (title, width))
            # # minimize
            # win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
            # # bring to forward
            # win32gui.SetForegroundWindow(hwnd)


if __name__ == '__main__':
    start_time = time.time()
    while True:
        EnumWindows(kill_window, 0)
        running_time = time.time() - start_time
        print('\r', "'kill_window.py' is running, in %.2f seconds: " % running_time, end='', flush=True)
        time.sleep(3)

