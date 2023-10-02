import win32gui
import win32api
import win32con
import win32ui
import numpy as np
import cv2


def grab_screen(display_index=None, region=None):
    # 获取系统上的显示器列表
    monitors = win32api.EnumDisplayMonitors()

    # 如果提供了区域参数，则根据区域参数裁剪屏幕内容
    left, top, x2, y2 = region
    width = x2 - left + 1
    height = y2 - top + 1

    # 根据传入的显示器索引选择特定的显示器
    monitor_info = monitors[display_index]  # 默认选择第一个显示器
    left_offset = monitor_info[2][0]
    top_offset = monitor_info[2][1]
    left += left_offset
    top += top_offset

    # 获取桌面窗口的句柄
    hwin = win32gui.GetDesktopWindow()
    # 获取桌面窗口的设备上下文
    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    # 复制屏幕内容到位图中
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)

    # 从位图中获取图像数据
    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.frombuffer(signedIntsArray, dtype='uint8')
    img.shape = (height, width, 4)

    # 清理资源
    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    # 将图像从 BGRA 格式转换为 RGB 格式
    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
