#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
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

std::ofstream g_logFile;
std::mutex g_mutex;
std::atomic<bool> g_running(true);
HHOOK g_keyboardHook = nullptr;
HHOOK g_mouseHook = nullptr;
bool g_cursorVisible = true;
bool g_isClipped = false;
int g_lastMouseX = 0;
int g_lastMouseY = 0;
bool g_initialMousePosSet = false;

inline int64_t GetHighResTimestamp() {
    FILETIME ft;
    GetSystemTimePreciseAsFileTime(&ft);
    return (static_cast<int64_t>(ft.dwHighDateTime) << 32) | ft.dwLowDateTime;
}

std::string KeyCodeToToken(DWORD vkCode, bool isExtended) {
    switch (vkCode) {
    case VK_LBUTTON: return "LB";
    case VK_RBUTTON: return "RB";
    case VK_MBUTTON: return "MB";
    case VK_XBUTTON1: return "XB1";
    case VK_XBUTTON2: return "XB2";
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
    case VK_ESCAPE: return "Esc";
    case VK_TAB: return "Tab";
    case VK_CAPITAL: return "Caps";
    case VK_SHIFT: return "Shift";
    case VK_CONTROL: return "Ctrl";
    case VK_MENU: return "Alt";
    case VK_SPACE: return "Space";
    case VK_BACK: return "Back";
    case VK_RETURN: return "Enter";
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

void LogMouseButton(int64_t timestamp, const char* event) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_logFile.is_open()) {
        g_logFile << timestamp << ",MOUSE," << event << "\n";
    }
}

void LogMouseWheel(int64_t timestamp, int delta) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_logFile.is_open()) {
        g_logFile << timestamp << ",MOUSE,WHEEL," << delta << "\n";
    }
}

void LogMouseAbs(int64_t timestamp, int x, int y) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_logFile.is_open()) {
        g_logFile << timestamp << ",MOUSE_ABS," << x << "," << y << "\n";
    }
}

void LogMouseRel(int64_t timestamp, int dx, int dy) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_logFile.is_open()) {
        g_logFile << timestamp << ",MOUSE_REL," << dx << "," << dy << "\n";
    }
}

LRESULT CALLBACK KeyboardHookProc(int nCode, WPARAM wParam, LPARAM lParam) {
    if (nCode >= 0) {
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

LRESULT CALLBACK MouseHookProc(int nCode, WPARAM wParam, LPARAM lParam) {
    if (nCode >= 0) {
        MSLLHOOKSTRUCT* pMouseStruct = (MSLLHOOKSTRUCT*)lParam;
        int64_t timestamp = GetHighResTimestamp();
        
        switch (wParam) {
        case WM_MOUSEMOVE: {
            int x = pMouseStruct->pt.x;
            int y = pMouseStruct->pt.y;
            
            if (g_isClipped) {
                int dx = x - g_lastMouseX;
                int dy = y - g_lastMouseY;
                if (dx != 0 || dy != 0) {
                    LogMouseRel(timestamp, dx, dy);
                }
            }
            g_lastMouseX = x;
            g_lastMouseY = y;
            break;
        }
        case WM_LBUTTONDOWN: LogMouseButton(timestamp, "LB_DOWN"); break;
        case WM_LBUTTONUP: LogMouseButton(timestamp, "LB_UP"); break;
        case WM_RBUTTONDOWN: LogMouseButton(timestamp, "RB_DOWN"); break;
        case WM_RBUTTONUP: LogMouseButton(timestamp, "RB_UP"); break;
        case WM_MBUTTONDOWN: LogMouseButton(timestamp, "MB_DOWN"); break;
        case WM_MBUTTONUP: LogMouseButton(timestamp, "MB_UP"); break;
        case WM_XBUTTONDOWN: {
            DWORD fwButton = HIWORD(pMouseStruct->mouseData);
            LogMouseButton(timestamp, fwButton == 1 ? "XB1_DOWN" : "XB2_DOWN");
            break;
        }
        case WM_XBUTTONUP: {
            DWORD fwButton = HIWORD(pMouseStruct->mouseData);
            LogMouseButton(timestamp, fwButton == 1 ? "XB1_UP" : "XB2_UP");
            break;
        }
        case WM_MOUSEWHEEL: {
            short delta = HIWORD(pMouseStruct->mouseData);
            LogMouseWheel(timestamp, delta);
            break;
        }
        case WM_MOUSEHWHEEL: {
            short delta = HIWORD(pMouseStruct->mouseData);
            LogMouseWheel(timestamp, delta);
            break;
        }
        }
    }
    return CallNextHookEx(g_mouseHook, nCode, wParam, lParam);
}

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

BOOL WINAPI ConsoleCtrlHandler(DWORD dwCtrlType) {
    if (dwCtrlType == CTRL_C_EVENT || dwCtrlType == CTRL_BREAK_EVENT) {
        std::cout << "\nShutting down..." << std::endl;
        g_running = false;
        return TRUE;
    }
    return FALSE;
}

int main(int argc, char* argv[]) {
    std::cout << "======================================" << std::endl;
    std::cout << "  KeyRecorder - Keyboard/Mouse Logger" << std::endl;
    std::cout << "======================================" << std::endl;
    std::cout << std::endl;

    std::string outputFile = "input_log.txt";
    if (argc > 1) {
        outputFile = argv[1];
    }

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

    SetConsoleCtrlHandler(ConsoleCtrlHandler, TRUE);

    g_logFile.open(outputFile, std::ios::out);
    if (!g_logFile.is_open()) {
        std::cerr << "Failed to open output file: " << outputFile << std::endl;
        return 1;
    }

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

    HINSTANCE hInstance = GetModuleHandle(NULL);

    g_keyboardHook = SetWindowsHookEx(WH_KEYBOARD_LL, KeyboardHookProc, hInstance, 0);
    if (!g_keyboardHook) {
        std::cerr << "Failed to install keyboard hook: " << GetLastError() << std::endl;
    } else {
        std::cout << "Keyboard hook installed successfully" << std::endl;
    }

    g_mouseHook = SetWindowsHookEx(WH_MOUSE_LL, MouseHookProc, hInstance, 0);
    if (!g_mouseHook) {
        std::cerr << "Failed to install mouse hook: " << GetLastError() << std::endl;
    } else {
        std::cout << "Mouse hook installed successfully" << std::endl;
    }

    POINT cursorPos;
    if (GetCursorPos(&cursorPos)) {
        g_lastMouseX = cursorPos.x;
        g_lastMouseY = cursorPos.y;
    }

    std::cout << "Recording started... (Press ESC to stop)" << std::endl;

    MSG msg;
    DWORD lastCursorCheck = GetTickCount();
    DWORD lastMousePoll = GetTickCount();

    while (g_running) {
        if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        DWORD now = GetTickCount();

        if (now - lastMousePoll >= 1) {
            if (!g_isClipped) {
                POINT pos;
                if (GetCursorPos(&pos)) {
                    int64_t timestamp = GetHighResTimestamp();
                    int dx = pos.x - g_lastMouseX;
                    int dy = pos.y - g_lastMouseY;
                    if (dx != 0 || dy != 0) {
                        LogMouseAbs(timestamp, pos.x, pos.y);
                    }
                    g_lastMouseX = pos.x;
                    g_lastMouseY = pos.y;
                }
            }
            lastMousePoll = now;
        }

        if (now - lastCursorCheck >= 100) {
            CheckCursorState(GetHighResTimestamp());
            lastCursorCheck = now;
        }
        Sleep(1);
    }

    std::cout << "Cleaning up..." << std::endl;

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
    
    std::cout << "Recording stopped. Log saved to: " << outputFile << std::endl;
    
    return 0;
}
