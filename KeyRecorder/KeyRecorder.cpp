#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <dinput.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <sstream>
#include <iomanip>

#pragma comment(lib, "dinput8.lib")
#pragma comment(lib, "dxguid.lib")
#pragma comment(lib, "winmm.lib")

// Global state
std::ofstream g_logFile;
std::mutex g_mutex;
std::atomic<bool> g_running(true);
HHOOK g_keyboardHook = nullptr;
HHOOK g_mouseHook = nullptr;
LPDIRECTINPUT8 g_directInput = nullptr;
LPDIRECTINPUTDEVICE8 g_mouseDevice = nullptr;
bool g_cursorVisible = true;
RECT g_clipRect = {0, 0, 0, 0};
bool g_isClipped = false;

void CheckCursorState(int64_t timestamp);

inline int64_t GetHighResTimestamp() {
    FILETIME ft;
    GetSystemTimePreciseAsFileTime(&ft);
    return (static_cast<int64_t>(ft.dwHighDateTime) << 32) | ft.dwLowDateTime;
}

// Key code to string mapping (matching Lumine's token format)
std::string KeyCodeToToken(DWORD vkCode, bool isExtended) {
    switch (vkCode) {
    // Mouse buttons
    case VK_LBUTTON: return "LB";
    case VK_RBUTTON: return "RB";
    case VK_MBUTTON: return "MB";
    case VK_XBUTTON1: return "XB1";
    case VK_XBUTTON2: return "XB2";

    // Numbers (1-9, 0)
    case '0': return "zero";
    case '1': return "one";
    case '2': return "two";
    case '3': return "three";
    case '4': return "four";
    case '5': return "five";
    case '6': return "six";
    case '7': return "seven";
    case '8': return "eight";
    case '9': return "nine";

    // Letters (A-Z)
    case 'A': return "A";
    case 'B': return "B";
    case 'C': return "C";
    case 'D': return "D";
    case 'E': return "E";
    case 'F': return "F";
    case 'G': return "G";
    case 'H': return "H";
    case 'I': return "I";
    case 'J': return "J";
    case 'K': return "K";
    case 'L': return "L";
    case 'M': return "M";
    case 'N': return "N";
    case 'O': return "O";
    case 'P': return "P";
    case 'Q': return "Q";
    case 'R': return "R";
    case 'S': return "S";
    case 'T': return "T";
    case 'U': return "U";
    case 'V': return "V";
    case 'W': return "W";
    case 'X': return "X";
    case 'Y': return "Y";
    case 'Z': return "Z";

    // Function keys
    case VK_F1: return "One";
    case VK_F2: return "Two";
    case VK_F3: return "Three";
    case VK_F4: return "Four";
    case VK_F5: return "Five";
    case VK_F6: return "Six";
    case VK_F7: return "Seven";
    case VK_F8: return "Eight";
    case VK_F9: return "Nine";
    case VK_F10: return "Ten";
    case VK_F11: return "Eleven";
    case VK_F12: return "Twelve";

    // Special keys
    case VK_ESCAPE: return "Esc";
    case VK_TAB: return "Tab";
    case VK_CAPITAL: return "Caps";
    case VK_SHIFT: return "Shift";
    case VK_CONTROL: return "Ctrl";
    case VK_MENU: return "Alt";
    case VK_SPACE: return "Space";
    case VK_BACK: return "Back";
    case VK_RETURN: return "Enter";

    // Navigation
    case VK_LEFT: return "Left";
    case VK_UP: return "Up";
    case VK_RIGHT: return "Right";
    case VK_DOWN: return "Down";
    case VK_HOME: return "Home";
    case VK_END: return "End";
    case VK_PRIOR: return "PgUp";
    case VK_NEXT: return "PgDn";
    case VK_INSERT: return "Ins";
    case VK_DELETE: return "Del";

    // Numpad
    case VK_NUMPAD0: return "Num0";
    case VK_NUMPAD1: return "Num1";
    case VK_NUMPAD2: return "Num2";
    case VK_NUMPAD3: return "Num3";
    case VK_NUMPAD4: return "Num4";
    case VK_NUMPAD5: return "Num5";
    case VK_NUMPAD6: return "Num6";
    case VK_NUMPAD7: return "Num7";
    case VK_NUMPAD8: return "Num8";
    case VK_NUMPAD9: return "Num9";
    case VK_MULTIPLY: return "Num*";
    case VK_ADD: return "Num+";
    case VK_SUBTRACT: return "Num-";
    case VK_DECIMAL: return "Num.";
    case VK_DIVIDE: return "Num/";

    default: return "";
    }
}

// Convert scan code to virtual key for extended keys
DWORD GetVirtualKeyCode(WPARAM wParam, LPARAM lParam) {
    // For most keys, wParam is the virtual key code
    return wParam;
}

// Keyboard hook procedure
LRESULT CALLBACK KeyboardHookProc(int nCode, WPARAM wParam, LPARAM lParam) {
    if (nCode >= 0) {
        // Check for ESC to stop recording
        if (wParam == VK_ESCAPE) {
            g_running = false;
            return CallNextHookEx(g_keyboardHook, nCode, wParam, lParam);
        }
        
        bool isDown = (wParam == WM_KEYDOWN || wParam == WM_SYSKEYDOWN);
        bool isUp = (wParam == WM_KEYUP || wParam == WM_SYSKEYUP);
        
        if (isDown || isUp) {
            KBDLLHOOKSTRUCT* kbStruct = (KBDLLHOOKSTRUCT*)lParam;
            DWORD vkCode = kbStruct->vkCode;
            bool isExtended = (kbStruct->flags & LLKHF_EXTENDED) != 0;

            std::string token = KeyCodeToToken(vkCode, isExtended);
            if (!token.empty()) {
                int64_t timestamp = GetHighResTimestamp();
                
                std::lock_guard<std::mutex> lock(g_mutex);
                if (g_logFile.is_open()) {
                    g_logFile << timestamp << ",KEY," << (isDown ? "DOWN" : "UP") << "," << token << "\n";
                    g_logFile.flush();
                }
            }
        }
    }
    return CallNextHookEx(g_keyboardHook, nCode, wParam, lParam);
}

// Mouse hook procedure (absolute position)
LRESULT CALLBACK MouseHookProc(int nCode, WPARAM wParam, LPARAM lParam) {
    if (nCode >= 0) {
        int64_t timestamp = GetHighResTimestamp();
        
        switch (wParam) {
        case WM_MOUSEMOVE: {
            CheckCursorState(timestamp);
            MSLLHOOKSTRUCT* pMouseStruct = (MSLLHOOKSTRUCT*)lParam;
            std::lock_guard<std::mutex> lock(g_mutex);
            if (g_logFile.is_open()) {
                g_logFile << timestamp << ",MOUSE_ABS," << pMouseStruct->pt.x << "," << pMouseStruct->pt.y << "\n";
            }
            break;
        }
        case WM_LBUTTONDOWN: {
            CheckCursorState(timestamp);
            std::lock_guard<std::mutex> lock(g_mutex);
            if (g_logFile.is_open()) {
                g_logFile << timestamp << ",MOUSE,LB_DOWN\n";
            }
            break;
        }
        case WM_LBUTTONUP: {
            std::lock_guard<std::mutex> lock(g_mutex);
            if (g_logFile.is_open()) {
                g_logFile << timestamp << ",MOUSE,LB_UP\n";
            }
            break;
        }
        case WM_RBUTTONDOWN: {
            std::lock_guard<std::mutex> lock(g_mutex);
            if (g_logFile.is_open()) {
                g_logFile << timestamp << ",MOUSE,RB_DOWN\n";
            }
            break;
        }
        case WM_RBUTTONUP: {
            std::lock_guard<std::mutex> lock(g_mutex);
            if (g_logFile.is_open()) {
                g_logFile << timestamp << ",MOUSE,RB_UP\n";
            }
            break;
        }
        case WM_MBUTTONDOWN: {
            std::lock_guard<std::mutex> lock(g_mutex);
            if (g_logFile.is_open()) {
                g_logFile << timestamp << ",MOUSE,MB_DOWN\n";
            }
            break;
        }
        case WM_MBUTTONUP: {
            std::lock_guard<std::mutex> lock(g_mutex);
            if (g_logFile.is_open()) {
                g_logFile << timestamp << ",MOUSE,MB_UP\n";
            }
            break;
        }
        case WM_MOUSEWHEEL: {
            MSLLHOOKSTRUCT* pMouseStruct = (MSLLHOOKSTRUCT*)lParam;
            short delta = HIWORD(pMouseStruct->mouseData);
            std::lock_guard<std::mutex> lock(g_mutex);
            if (g_logFile.is_open()) {
                g_logFile << timestamp << ",MOUSE,WHEEL," << delta << "\n";
            }
            break;
        }
        case WM_MOUSEHWHEEL: {
            MSLLHOOKSTRUCT* pMouseStruct = (MSLLHOOKSTRUCT*)lParam;
            short delta = HIWORD(pMouseStruct->mouseData);
            std::lock_guard<std::mutex> lock(g_mutex);
            if (g_logFile.is_open()) {
                g_logFile << timestamp << ",MOUSE,HWHEEL," << delta << "\n";
            }
            break;
        }
        }
    }
    return CallNextHookEx(g_mouseHook, nCode, wParam, lParam);
}

// DirectInput mouse polling thread (relative movement)
void PollRelativeMouse() {
    DIDEVICEOBJECTDATA mouseData[64];
    DWORD dwElements = 64;
    HRESULT hr;

    while (g_running) {
        if (g_mouseDevice) {
            dwElements = 64;
            hr = g_mouseDevice->GetDeviceData(sizeof(DIDEVICEOBJECTDATA), mouseData, &dwElements, 0);
            
            if (SUCCEEDED(hr) && dwElements > 0) {
                int64_t timestamp = GetHighResTimestamp();
                int relX = 0, relY = 0;
                
                for (DWORD i = 0; i < dwElements; i++) {
                    switch (mouseData[i].dwOfs) {
                    case DIMOFS_X:
                        relX += mouseData[i].dwData;
                        break;
                    case DIMOFS_Y:
                        relY += mouseData[i].dwData;
                        break;
                    case DIMOFS_Z:
                        // Mouse wheel (relative)
                        std::lock_guard<std::mutex> lock(g_mutex);
                        if (g_logFile.is_open()) {
                            g_logFile << timestamp << ",MOUSE_REL,WHEEL," << mouseData[i].dwData << "\n";
                        }
                        break;
                    }
                }

                if (relX != 0 || relY != 0) {
                    std::lock_guard<std::mutex> lock(g_mutex);
                    if (g_logFile.is_open()) {
                        g_logFile << timestamp << ",MOUSE_REL," << relX << "," << relY << "\n";
                    }
                }
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5)); // 200Hz polling
    }
}

// Initialize DirectInput
bool InitDirectInput(HINSTANCE hInstance) {
    HRESULT hr = DirectInput8Create(hInstance, DIRECTINPUT_VERSION, IID_IDirectInput8, (LPVOID*)&g_directInput, NULL);
    if (FAILED(hr)) {
        std::cerr << "Failed to create DirectInput object" << std::endl;
        return false;
    }

    // Create mouse device
    hr = g_directInput->CreateDevice(GUID_SysMouse, &g_mouseDevice, NULL);
    if (FAILED(hr)) {
        std::cerr << "Failed to create mouse device" << std::endl;
        return false;
    }

    // Set data format
    hr = g_mouseDevice->SetDataFormat(&c_dfDIMouse2);
    if (FAILED(hr)) {
        std::cerr << "Failed to set mouse data format" << std::endl;
        return false;
    }

    // Set cooperative level (exclusive if foreground, otherwise non-exclusive)
    HWND hwnd = GetForegroundWindow();
    if (!hwnd) hwnd = GetConsoleWindow();
    hr = g_mouseDevice->SetCooperativeLevel(hwnd, DISCL_NONEXCLUSIVE | DISCL_BACKGROUND);
    if (FAILED(hr)) {
        std::cerr << "Failed to set cooperative level" << std::endl;
        return false;
    }

    // Set relative mouse mode BEFORE acquiring
    DIPROPDWORD dipdw;
    dipdw.diph.dwSize = sizeof(DIPROPDWORD);
    dipdw.diph.dwHeaderSize = sizeof(DIPROPHEADER);
    dipdw.diph.dwObj = 0;
    dipdw.diph.dwHow = DIPH_DEVICE;
    dipdw.dwData = DIPROPAXISMODE_REL;
    hr = g_mouseDevice->SetProperty(DIPROP_AXISMODE, &dipdw.diph);
    if (FAILED(hr)) {
        std::cerr << "Warning: Failed to set relative mouse mode" << std::endl;
    }

    // Acquire the device
    hr = g_mouseDevice->Acquire();
    if (FAILED(hr)) {
        std::cerr << "Failed to acquire mouse device" << std::endl;
    }

    return true;
}

// Check and log cursor state changes
void CheckCursorState(int64_t timestamp) {
    CURSORINFO ci = {sizeof(CURSORINFO)};
    if (GetCursorInfo(&ci)) {
        bool visible = (ci.flags & (CURSOR_SHOWING | CURSOR_SUPPRESSED)) != 0;
        if (visible != g_cursorVisible) {
            g_cursorVisible = visible;
            std::lock_guard<std::mutex> lock(g_mutex);
            if (g_logFile.is_open()) {
                g_logFile << timestamp << ",MOUSE," << (visible ? "SHOW" : "HIDE") << "\n";
            }
        }
    }
    
    RECT clipRect;
    if (GetClipCursor(&clipRect)) {
        int screenWidth = GetSystemMetrics(SM_CXSCREEN);
        int screenHeight = GetSystemMetrics(SM_CYSCREEN);
        int clipWidth = clipRect.right - clipRect.left;
        int clipHeight = clipRect.bottom - clipRect.top;
        bool clipped = (clipWidth < screenWidth || clipHeight < screenHeight);
        if (clipped != g_isClipped) {
            g_isClipped = clipped;
            std::lock_guard<std::mutex> lock(g_mutex);
            if (g_logFile.is_open()) {
                g_logFile << timestamp << ",MOUSE," << (clipped ? "LOCK" : "UNLOCK") << "\n";
            }
        }
    } else if (g_isClipped) {
        g_isClipped = false;
        std::lock_guard<std::mutex> lock(g_mutex);
        if (g_logFile.is_open()) {
            g_logFile << timestamp << ",MOUSE,UNLOCK\n";
        }
    }
}

// Cleanup
void Cleanup() {
    g_running = false;

    if (g_mouseDevice) {
        g_mouseDevice->Unacquire();
        g_mouseDevice->Release();
        g_mouseDevice = nullptr;
    }

    if (g_directInput) {
        g_directInput->Release();
        g_directInput = nullptr;
    }

    if (g_keyboardHook) {
        UnhookWindowsHookEx(g_keyboardHook);
        g_keyboardHook = nullptr;
    }

    if (g_mouseHook) {
        UnhookWindowsHookEx(g_mouseHook);
        g_mouseHook = nullptr;
    }

    if (g_logFile.is_open()) {
        g_logFile.close();
    }
}

// Console control handler
BOOL WINAPI ConsoleCtrlHandler(DWORD dwCtrlType) {
    if (dwCtrlType == CTRL_C_EVENT || dwCtrlType == CTRL_BREAK_EVENT) {
        std::cout << "\nShutting down..." << std::endl;
        Cleanup();
        exit(0);
        return TRUE;
    }
    return FALSE;
}

int main(int argc, char* argv[]) {
    std::cout << "======================================" << std::endl;
    std::cout << "  KeyRecorder - Keyboard/Mouse Logger" << std::endl;
    std::cout << "======================================" << std::endl;
    std::cout << std::endl;

    // Get output filename
    std::string outputFile = "input_log.txt";
    if (argc > 1) {
        outputFile = argv[1];
    }

    // Generate timestamp for filename if not provided
    if (outputFile == "input_log.txt") {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time), "%Y%m%d_%H%M%S");
        outputFile = "input_log_" + ss.str() + ".txt";
    }

    std::cout << "Output file: " << outputFile << std::endl;
    std::cout << std::endl;
    std::cout << "Controls:" << std::endl;
    std::cout << "  Press ESC to stop recording" << std::endl;
    std::cout << std::endl;

    // Set console control handler
    SetConsoleCtrlHandler(ConsoleCtrlHandler, TRUE);

    // Open log file
    g_logFile.open(outputFile, std::ios::out);
    if (!g_logFile.is_open()) {
        std::cerr << "Failed to open output file: " << outputFile << std::endl;
        return 1;
    }

    // Write header
    g_logFile << "# KeyRecorder Input Log" << std::endl;
    g_logFile << "# Format: timestamp,EVENT_TYPE,data" << std::endl;
    g_logFile << "# timestamp: Windows FILETIME (100-nanosecond intervals since 1601-01-01)" << std::endl;
    g_logFile << "#" << std::endl;
    g_logFile << "# Events:" << std::endl;
    g_logFile << "#   KEY,DOWN|UP,token" << std::endl;
    g_logFile << "#   MOUSE_ABS,x,y" << std::endl;
    g_logFile << "#   MOUSE_REL,dx,dy" << std::endl;
    g_logFile << "#   MOUSE,WHEEL,delta" << std::endl;
    g_logFile << "#   MOUSE,SHOW|HIDE" << std::endl;
    g_logFile << "#   MOUSE,LOCK|UNLOCK" << std::endl;
    g_logFile << "#" << std::endl;
    g_logFile.flush();

    // Get module handle
    HINSTANCE hInstance = GetModuleHandle(NULL);

    // Initialize DirectInput
    if (!InitDirectInput(hInstance)) {
        std::cerr << "Warning: DirectInput initialization failed, relative mouse may not work" << std::endl;
    }

    // Install keyboard hook
    g_keyboardHook = SetWindowsHookEx(WH_KEYBOARD_LL, KeyboardHookProc, hInstance, 0);
    if (!g_keyboardHook) {
        std::cerr << "Failed to install keyboard hook: " << GetLastError() << std::endl;
    } else {
        std::cout << "Keyboard hook installed successfully" << std::endl;
    }

    // Install mouse hook
    g_mouseHook = SetWindowsHookEx(WH_MOUSE_LL, MouseHookProc, hInstance, 0);
    if (!g_mouseHook) {
        std::cerr << "Failed to install mouse hook: " << GetLastError() << std::endl;
    } else {
        std::cout << "Mouse hook installed successfully" << std::endl;
    }

    // Start relative mouse polling thread
    std::thread mousePollThread(PollRelativeMouse);

    std::cout << std::endl;
    std::cout << "Recording started... (Press ESC to stop)" << std::endl;

    // Message loop
    MSG msg;
    DWORD lastCursorCheck = GetTickCount();
    while (g_running) {
        if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        } else {
            DWORD now = GetTickCount();
            if (now - lastCursorCheck >= 100) {
                CheckCursorState(GetHighResTimestamp());
                lastCursorCheck = now;
            }
            Sleep(1); // Prevent busy-waiting
        }
    }

    // Cleanup
    std::cout << "Cleaning up..." << std::endl;
    
    if (mousePollThread.joinable()) {
        mousePollThread.join();
    }
    
    Cleanup();
    
    std::cout << "Recording stopped. Log saved to: " << outputFile << std::endl;
    
    return 0;
}
