"""Minimal Vulkan probe: call vkCreateInstance via ctypes and enumerate devices.

This reproduces the exact call that failed inside Singularity on Leonardo boost nodes
(G5b / D25: VK_ERROR_INCOMPATIBLE_DRIVER, journal 2026-08-19), with no dependency on
vulkaninfo being installed in the image. rc 0 = VK_SUCCESS, rc -9 = INCOMPATIBLE_DRIVER.

Exit code: 0 iff an instance was created AND >=1 physical device enumerates.
"""

import ctypes
import sys

VK_API_VERSION_1_1 = (1 << 22) | (1 << 12)  # VK_MAKE_API_VERSION(0, 1, 1, 0)


class VkApplicationInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_int),  # VK_STRUCTURE_TYPE_APPLICATION_INFO = 0
        ("pNext", ctypes.c_void_p),
        ("pApplicationName", ctypes.c_char_p),
        ("applicationVersion", ctypes.c_uint32),
        ("pEngineName", ctypes.c_char_p),
        ("engineVersion", ctypes.c_uint32),
        ("apiVersion", ctypes.c_uint32),
    ]


class VkInstanceCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_int),  # VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("pApplicationInfo", ctypes.c_void_p),
        ("enabledLayerCount", ctypes.c_uint32),
        ("ppEnabledLayerNames", ctypes.c_void_p),
        ("enabledExtensionCount", ctypes.c_uint32),
        ("ppEnabledExtensionNames", ctypes.c_void_p),
    ]


def main() -> int:
    try:
        vk = ctypes.CDLL("libvulkan.so.1")
    except OSError as e:
        print(f"VK_PROBE: libvulkan.so.1 load FAILED: {e}")
        return 1

    app = VkApplicationInfo(0, None, b"vk_probe", 0, b"none", 0, VK_API_VERSION_1_1)
    ci = VkInstanceCreateInfo(
        1, None, 0, ctypes.cast(ctypes.byref(app), ctypes.c_void_p), 0, None, 0, None
    )
    inst = ctypes.c_void_p()
    rc = vk.vkCreateInstance(ctypes.byref(ci), None, ctypes.byref(inst))
    print(f"VK_PROBE: vkCreateInstance rc={rc} (0=SUCCESS, -9=ERROR_INCOMPATIBLE_DRIVER)")
    if rc != 0:
        return 1

    count = ctypes.c_uint32(0)
    vk.vkEnumeratePhysicalDevices(inst, ctypes.byref(count), None)
    print(f"VK_PROBE: physical devices: {count.value}")
    if count.value == 0:
        return 1

    devs = (ctypes.c_void_p * count.value)()
    vk.vkEnumeratePhysicalDevices(inst, ctypes.byref(count), devs)
    for i, dev in enumerate(devs):
        # VkPhysicalDeviceProperties: deviceName (256 chars) sits at byte offset 20,
        # after apiVersion/driverVersion/vendorID/deviceID/deviceType (5 x uint32).
        buf = ctypes.create_string_buffer(4096)
        vk.vkGetPhysicalDeviceProperties(ctypes.c_void_p(dev), buf)
        name = buf.raw[20 : 20 + 256].split(b"\x00", 1)[0].decode(errors="replace")
        api = int.from_bytes(buf.raw[0:4], "little")
        print(
            f"VK_PROBE: device {i}: {name} "
            f"(api {api >> 22 & 0x7F}.{api >> 12 & 0x3FF}.{api & 0xFFF})"
        )
    print("VK_PROBE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
