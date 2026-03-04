import os
import time
import http.cookiejar
from playwright.sync_api import sync_playwright

def get_cookies_for_playwright(cookie_file):
    """
    将 Netscape 格式 (cookies.txt) 转换为 Playwright 需要的 JSON 格式
    """
    cj = http.cookiejar.MozillaCookieJar(cookie_file)
    try:
        cj.load()
        cookies = []
        for c in cj:
            # Playwright 需要 domain, name, value, path 等字段
            cookie_dict = {
                'name': c.name,
                'value': c.value,
                'domain': c.domain,
                'path': c.path,
                'secure': c.secure,
                'httpOnly': False  # 通常设置为 False 即可
            }
            cookies.append(cookie_dict)
        return cookies
    except Exception as e:
        print(f"[Cookie Error] Failed to parse cookie file: {e}")
        return []

def download_ieee_with_browser(doi_url, cookie_path="coockies\\ieee.txt", save_dir="pdfs"):
    """
    使用 Playwright 模拟浏览器下载 IEEE 论文
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"[Browser] Starting download for DOI: {doi_url}")

    with sync_playwright() as p:
        # 启动浏览器 (headless=True 表示不显示界面，调试时可改为 False)
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        )

        # 1. 注入 Cookie
        cookies = get_cookies_for_playwright(cookie_path)
        if not cookies:
            print("[Error] No cookies found. Aborting.")
            return None
        context.add_cookies(cookies)

        page = context.new_page()

        try:
            # 2. 访问 DOI (让 IEEE 进行重定向)
            print("[Browser] Navigating to paper page...")
            page.goto(doi_url, timeout=30000)
            
            # 等待页面加载，IEEE 可能会跳转多次
            page.wait_for_load_state("networkidle")

            # 3. 寻找 PDF 按钮并触发下载
            # IEEE 的 PDF 按钮通常包含 "pdf" 字样或特定的 class
            # 我们直接构造 stamp URL 可能更稳，但先尝试从页面提取
            
            # 获取当前的 arnumber (文章ID)
            current_url = page.url
            import re
            arnumber = None
            match = re.search(r"document/(\d+)", current_url)
            if match:
                arnumber = match.group(1)
                print(f"[Browser] Detected arnumber: {arnumber}")
                
                # 构造直接下载链接，这通常会触发浏览器的下载事件
                pdf_link = f"https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber={arnumber}"
                
                print(f"[Browser] Triggering PDF download via: {pdf_link}")
                
                # 监听下载事件
                with page.expect_download(timeout=60000) as download_info:
                    # 访问 stamp 链接，这会触发 JS 重定向并最终开始下载
                    page.goto(pdf_link)
                
                download = download_info.value
                # 使用文章标题或 ID 命名
                suggested_filename = download.suggested_filename
                save_path = os.path.join(save_dir, suggested_filename)
                
                download.save_as(save_path)
                print(f"✅ [Success] Downloaded: {save_path}")
                browser.close()
                return save_path
            else:
                print("[Error] Could not find arnumber in URL. Are you logged in?")
                # 截图方便调试
                page.screenshot(path="debug_error.png")
                
        except Exception as e:
            print(f"❌ [Error] Playwright download failed: {e}")
            # 出错时截图
            try:
                page.screenshot(path="debug_exception.png")
                print("   Screenshot saved to debug_exception.png")
            except:
                pass
            
        browser.close()
    return None

# # --- 测试代码 ---
# if __name__ == "__main__":
#     # 使用你刚才失败的那个 DOI 进行测试
#     test_doi = "https://doi.org/10.1109/TCCN.2017.2758370"
#     # 确保目录下有 ieee_cookies.txt
#     download_ieee_with_browser(test_doi)


# import os
# from playwright.sync_api import sync_playwright

def setup_login():
    # 这里定义一个持久化的用户数据目录
    user_data_dir = os.path.abspath("ieee_user_data")
    
    with sync_playwright() as p:
        print(f"正在启动浏览器，数据存储在: {user_data_dir}")
        print("请在弹出的浏览器窗口中，手动完成 IEEE 的机构登录。")
        
        # 启动持久化上下文 (Persistent Context)
        # headless=False 意味着你会看到浏览器弹出
        context = p.chromium.launch_persistent_context(
            user_data_dir=user_data_dir,
            headless=False, 
            viewport={"width": 1280, "height": 720},
            # 伪装成正常的 Chrome 浏览器
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
        )
        
        page = context.new_page()
        page.goto("https://ieeexplore.ieee.org/Xplore/home.jsp")
        
        # --- 关键步骤 ---
        print("\n" + "="*50)
        print("【操作指南】")
        print("1. 浏览器已打开。")
        print("2. 请点击网页上方的 'Institutional Sign In'。")
        print("3. 搜索 'Southeast University' 或使用你的账号登录。")
        print("4. 确保右上角显示了你的学校名称 (Access provided by: Southeast University)。")
        print("5. 登录成功后，回到这个黑窗口按回车键保存。")
        print("="*50 + "\n")
        
        input("登录成功了吗？按回车键退出并保存状态...")
        
        context.close()
        print("登录状态已保存！现在可以使用下载脚本了。")

# if __name__ == "__main__":
#     setup_login()
    
import os
import re
from playwright.sync_api import sync_playwright

def download_paper_with_profile(doi_url, save_dir="pdfs"):
    user_data_dir = os.path.abspath("ieee_user_data") # 必须和刚才的路径一致
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"[Browser] 使用已保存的登录状态下载: {doi_url}")

    with sync_playwright() as p:
        # 注意：这里加载之前的 user_data_dir
        context = p.chromium.launch_persistent_context(
            user_data_dir=user_data_dir,
            headless=True, # 调试成功后可改为 True
            accept_downloads=True # 允许下载
        )
        
        page = context.pages[0] # 持久化上下文通常默认打开一个 tab
        
        try:
            print("[Browser] 正在跳转...")
            page.goto(doi_url, timeout=60000)
            page.wait_for_load_state("domcontentloaded")
            
            # 检查是否已登录 (右上角是否有 Access provided by)
            # 这一步是为了验证 Session 是否还在
            if "Institutional Sign In" in page.content():
                print("❌ [Error] 检测到未登录状态！Session 可能已过期。")
                print("   请重新运行 setup_login.py 更新状态。")
                # context.close()
                # return None

            # 提取 arnumber 并下载
            current_url = page.url
            match = re.search(r"document/(\d+)", current_url)
            
            if match:
                arnumber = match.group(1)
                pdf_link = f"https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber={arnumber}"
                print(f"[Browser] 触发下载: {pdf_link}")
                
                with page.expect_download(timeout=180000) as download_info:
                    page.goto(pdf_link)
                
                download = download_info.value
                save_path = os.path.join(save_dir, download.suggested_filename)
                download.save_as(save_path)
                
                print(f"✅ [Success] 文件已保存: {save_path}")
                context.close()
                return save_path
            else:
                print("[Error] 无法解析文章ID，可能页面加载不完整。")
                
        except Exception as e:
            print(f"❌ [Exception] {e}")
            context.close()
    
    return None

# if __name__ == "__main__":
#     # 使用你刚才失败的那个 DOI 测试
#     test_doi = "https://doi.org/10.1109/TCCN.2017.2758370"
#     download_paper_with_profile(test_doi)


import os
import re
import time
from playwright.sync_api import sync_playwright

class InteractiveIEEEDownloader:
    def __init__(self, save_dir="./pdfs_ieee"):
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        print("🚀 [Init] 正在启动浏览器...")
        self.playwright = sync_playwright().start()
        
        # 注意：headless=False 必须开启，因为需要你手动操作登录
        self.browser = self.playwright.chromium.launch(headless=False)
        self.context = self.browser.new_context(
            viewport={"width": 1280, "height": 800},
            accept_downloads=True # 允许下载
        )
        self.page = self.context.new_page()

        # --- 核心：人工登录环节 ---
        print("\n" + "="*50)
        print("【请执行以下步骤】")
        print("1. 浏览器窗口已打开，正在前往 IEEE Xplore...")
        self.page.goto("https://ieeexplore.ieee.org/Xplore/home.jsp")
        
        print("2. ⚠️ 请在浏览器中点击 'Institutional Sign In'。")
        print("3. 搜索 'Southeast University' 并完成统一身份认证。")
        print("4. 等待直到浏览器右上角显示 'Access provided by: Southeast University'。")
        print("="*50)
        
        # 程序在这里暂停，等待你按回车
        input("\n>>> 登录成功了吗？请在确认右上角有学校名后，在此处按回车键继续... <<<")
        print("✅ [Confirmed] Agent 接管浏览器，开始下载任务...")

    def download(self, doi):
        """
        混合模式下载：浏览器负责登录跳转，Requests 负责下载文件
        """
        print(f"🔍 [IEEE] 处理 DOI: {doi}")
        
        clean_name = doi.replace("https://doi.org/", "").replace("/", "_") + ".pdf"
        save_path = os.path.join(self.save_dir, clean_name)
        
        if os.path.exists(save_path):
            print(f"   ✅ 文件已存在: {save_path}")
            return save_path

        try:
            # 1. 浏览器访问，解决 SSO 跳转
            print("   ⏳ 正在访问链接 (允许 SSO 跳转)...")
            try:
                # 使用 wait_until="domcontentloaded" 只要页面骨架出来就行
                self.page.goto(doi, timeout=180000, wait_until="domcontentloaded")
            except Exception as e:
                pass # 忽略跳转中的中断错误

            # 2. 等待直到 URL 稳定 (确保跳到了 stamp.jsp 或具体的 PDF 链接)
            # 我们给它一点时间完成重定向
            self.page.wait_for_timeout(5000) 

            # 3. 获取当前 URL 和 arnumber
            current_url = self.page.url
            print(f"   📍 最终跳转地址: {current_url}")
            
            # 4. 提取文章 ID (arnumber)
            import re
            match = re.search(r"arnumber=(\d+)", current_url)
            if not match:
                 # 备选策略：有时候在 document/xxxx 路径下
                 match = re.search(r"document/(\d+)", current_url)
            
            if match:
                arnumber = match.group(1)
                # 构造最原始的 PDF 接口 (这个接口配合 Cookie 必得 PDF)
                pdf_url = f"https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber={arnumber}"
                print(f"   🎯 锁定 PDF 真实地址: {pdf_url}")

                # --- 关键步骤：Cookie 桥接 ---
                # 从 Playwright 浏览器中“偷”出当前的 Cookie
                browser_cookies = self.context.cookies()
                cookie_dict = {c['name']: c['value'] for c in browser_cookies}
                
                # 伪装 User-Agent (必须和浏览器一致)
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
                }

                # 使用 requests 带着浏览器的 Cookie 去下载
                import requests
                print("   ⬇️  正在通过 Requests 接管下载...")
                r = requests.get(pdf_url, cookies=cookie_dict, headers=headers, stream=True, timeout=60)
                
                # 检查是否真的拿到了 PDF (而不是 HTML)
                content_type = r.headers.get("Content-Type", "")
                if "application/pdf" in content_type:
                    with open(save_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print(f"   ✅ [Success] 下载成功: {save_path}")
                    return save_path
                else:
                    print(f"   ❌ 下载失败，服务器返回了: {content_type}")
                    # 如果返回的是 HTML，可能是 IEEE 的反爬或者是 iframe 嵌套
                    # 这种情况下，我们尝试解析 HTML 里的 iframe src
                    if "text/html" in content_type:
                        iframe_match = re.search(r'<iframe src="([^"]+)"', r.text)
                        if iframe_match:
                            real_pdf = iframe_match.group(1)
                            print(f"   🔄 发现嵌套 PDF，尝试二级下载: {real_pdf}")
                            r2 = requests.get(real_pdf, cookies=cookie_dict, headers=headers)
                            with open(save_path, "wb") as f:
                                f.write(r2.content)
                            print(f"   ✅ [Success] 二级下载成功: {save_path}")
                            return save_path

            else:
                print("   ❌ 无法解析 arnumber，可能还在登录页。")

        except Exception as e:
            print(f"   ❌ [Exception] {e}")

        return None

    def close(self):
        """任务结束后关闭"""
        print("👋 关闭浏览器资源")
        self.context.close()
        self.browser.close()
        self.playwright.stop()

# --- 使用示例 ---
# if __name__ == "__main__":
#     # 1. 初始化 (会弹出浏览器，请配合登录)
#     downloader = InteractiveIEEEDownloader()
    
#     # 2. 定义你要下载的列表 (模拟 Agent 思考出的结果)
#     paper_dois = [
#         "https://doi.org/10.1109/TCCN.2017.2758370"
#         "https://doi.org/10.1109/LSP.2023.3327872"
#     ]
    
#     # 3. 循环下载
#     try:
#         for doi in paper_dois:
#             downloader.download(doi)
#             time.sleep(2) # 稍微歇一下，防止被封
#     finally:
#         # 确保最后关闭浏览器
#         downloader.close()
        
        
import os
import requests
import re
from urllib.parse import urljoin

import os
import requests
import re
import urllib3

# --- 关键修改 1: 屏蔽烦人的 SSL 警告 ---
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class SciHubDownloader:
    def __init__(self, save_dir="./pdfs_scihub"):
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        # --- 关键修改 2: 扩充镜像列表 (按优先级排序) ---
        self.mirrors = [
            "https://sci-hub.se",   # 最常用
            "https://sci-hub.ru",   # 俄罗斯源，通常很稳
            "https://sci-hub.st",   # 备用
            "https://sci-hub.ee",   # 有时可用
            "https://sci-hub.wf"    # 备用
        ]
        
        # 伪装成普通浏览器，防止被拦截
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        }

    def download(self, doi):
        """
        下载主函数
        """
        # 1. 清理 DOI，提取纯净字符串
        clean_doi = doi.replace("https://doi.org/", "").replace("http://dx.doi.org/", "")
        
        # 2. 检查本地是否已存在
        filename = f"{clean_doi.replace('/', '_')}.pdf"
        save_path = os.path.join(self.save_dir, filename)
        if os.path.exists(save_path):
            print(f"✅ [Exist] 本地已存在: {filename}")
            return save_path

        print(f"🔍 [Sci-Hub] 开始搜索 DOI: {clean_doi}")

        # 3. 遍历镜像尝试下载
        for mirror in self.mirrors:
            target_url = f"{mirror}/{clean_doi}"
            try:
                # 请求页面 (verify=False 必须保留，但警告已被我们在开头屏蔽)
                response = requests.get(target_url, headers=self.headers, verify=False, timeout=15)
                print(f"{response.text}")
                # 如果页面返回 200 正常
                if response.status_code == 200:
                    # 尝试解析 PDF 链接
                    pdf_url = self._extract_pdf_link(response.text, mirror)
                    # open 'log.txt'
                    
                    print(f"{pdf_url}")
                    if pdf_url:
                        print(f"   ⬇️  [Downloading] 正在从 {mirror} 下载...")
                        
                        # 下载文件流
                        file_resp = requests.get(pdf_url, headers=self.headers, verify=False, timeout=60, stream=True)
                        
                        # 再次检查文件类型，防止下载到 HTML 报错页面
                        content_type = file_resp.headers.get("Content-Type", "").lower()
                        if "application/pdf" in content_type:
                            with open(save_path, "wb") as f:
                                for chunk in file_resp.iter_content(chunk_size=8192):
                                    f.write(chunk)
                            print(f"   ✅ [Success] 下载成功: {save_path}")
                            return save_path
                        else:
                            print(f"   ⚠️  [Warning] 链接指向的不是 PDF (Content-Type: {content_type})，尝试下一个镜像...")
                    else:
                        # 页面加载了，但没找到 PDF 按钮，可能是 Sci-Hub 还没收录这篇文章
                        if "article not found" in response.text.lower():
                            print(f"   ❌ [Not Found] Sci-Hub 暂未收录此文章。")
                            break # 其他镜像也是共用数据库，不用试了
                        
            except requests.exceptions.RequestException:
                # 网络连接报错 (比如 DNS 解析失败，或被校园网阻断)
                # print(f"   ⚠️  镜像 {mirror} 无法连接，跳过...")
                pass
            except Exception as e:
                print(f"   ⚠️  未知错误: {e}")
        
        print(f"❌ [Failed] 所有镜像均尝试失败: {clean_doi}")
        return None

    def _extract_pdf_link(self, html, base_url):
        """
        从 Sci-Hub 网页 HTML 中提取 PDF 真实下载地址
        """
        try:
            # 策略 A: 寻找 <embed> 或 <iframe> (Sci-Hub 标准结构)
            # 例如: <embed type="application/pdf" src="//sci-hub.se/downloads/xxxx/xxxx.pdf" id="pdf">
            match = re.search(r'<(?:embed|iframe) [^>]*src="([^"]+)"', html)
            if match:
                return self._fix_url(match.group(1), base_url)
            
            # 策略 B: 寻找 onclick 跳转 (Sci-Hub 备用结构)
            # 例如: location.href='//sci-hub.se/downloads/xxxx.pdf'
            match = re.search(r"location\.href='([^']+)'", html)
            if match:
                return self._fix_url(match.group(1), base_url)

            # 策略 C: 寻找 id="buttons" 里的 save 链接
            match = re.search(r'<div id="buttons">.*?<a href="#" onclick="location\.href=\'([^\']+)\'', html, re.DOTALL)
            if match:
                 return self._fix_url(match.group(1), base_url)

        except Exception:
            pass
        return None

    def _fix_url(self, url, base_url):
        """修复相对路径"""
        if url.startswith("//"):
            return "https:" + url
        if url.startswith("/"):
            return base_url + url
        return url

# # --- 测试用例 ---
# if __name__ == "__main__":
#     downloader = SciHubDownloader()
    
#     # 测试 1: 经典的 Deep Learning 论文 (应该能下)
#     test_doi_1 = "10.1109/TCCN.2017.2758370"
#     downloader.download(test_doi_1)

# # --- 使用示例 ---
# if __name__ == "__main__":
#     downloader = SciHubDownloader()
    
#     # 示例 DOI (一篇较旧的经典论文)
#     test_doi = "https://doi.org/10.1109/TCCN.2017.2758370"
#     downloader.download(test_doi)


import os
import requests
import re
import urllib3
from playwright.sync_api import sync_playwright

# 屏蔽 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class SciHubPlaywrightDownloader:
    def __init__(self, save_dir="./pdfs_scihub"):
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        # 常用镜像
        self.mirrors = [
            "https://sci-hub.se", 
            "https://sci-hub.ru", 
            "https://sci-hub.st"
        ]

    def download(self, doi):
        clean_doi = doi.replace("https://doi.org/", "").replace("http://dx.doi.org/", "")
        filename = f"{clean_doi.replace('/', '_')}.pdf"
        save_path = os.path.join(self.save_dir, filename)

        if os.path.exists(save_path):
            print(f"✅ [Exist] 文件已存在: {filename}")
            return save_path

        print(f"🔍 [Sci-Hub] 正在搜索 DOI: {clean_doi}")

        with sync_playwright() as p:
            # 建议使用 headless=True 后台运行。如果一直失败，可以改成 False 看看具体情况
            # 在你的 SciHubPlaywrightDownloader 类中，修改 launch 这一行：

            browser = p.chromium.launch(
                headless=True,
                # 填入你本地代理软件的端口，常见的是 7890 或 10809
                proxy={"server": "http://127.0.0.1:7897"} 
            )
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
            )
            page = context.new_page()

            for mirror in self.mirrors:
                target_url = f"{mirror}/{clean_doi}"
                try:
                    print(f"   🌐 正在访问镜像: {target_url}")
                    page.goto(target_url, timeout=30000)
                    page.wait_for_load_state("domcontentloaded")

                    # ==========================================
                    # 核心逻辑 1：自动处理 "Are you a robot?" 验证
                    # ==========================================
                    if page.locator(".answer").is_visible() or "are you are robot" in page.content().lower():
                        print("   🤖 触发人机验证，正在自动点击 'No' 进行破解...")
                        try:
                            # 点击那个 onclick="check()" 的 No 按钮
                            page.locator(".answer").click()
                            # 验证成功后网页会自动执行 window.location.reload()，所以我们要等待导航完成
                            page.wait_for_navigation(timeout=15000)
                            print("   🔓 验证破解成功，页面已刷新！")
                        except Exception as e:
                            print(f"   ⚠️ 验证破解超时或失败: {e}")
                            continue # 这个镜像废了，换下一个

                    # 如果页面显示 article not found
                    if "article not found" in page.content().lower():
                        print(f"   ❌ [Not Found] 此文章未被收录。")
                        break # 直接退出循环

                    # ==========================================
                    # 核心逻辑 2：提取真实的 PDF 地址
                    # ==========================================
                    pdf_url = None
                    # 查找 <embed id="pdf"> 或 <iframe id="pdf">
                    pdf_element = page.locator("iframe#pdf, embed#pdf").first
                    if pdf_element.count() > 0:
                        pdf_url = pdf_element.get_attribute("src")
                    
                    # 备用方案：查找左侧的 Save 按钮跳转链接
                    if not pdf_url:
                        html_content = page.content()
                        match = re.search(r"location\.href='([^']+)'", html_content)
                        if match:
                            pdf_url = match.group(1)

                    # ==========================================
                    # 核心逻辑 3：桥接 Cookie 并下载 PDF
                    # ==========================================
                    if pdf_url:
                        # 修复相对路径
                        if pdf_url.startswith("//"):
                            pdf_url = "https:" + pdf_url
                        elif pdf_url.startswith("/"):
                            pdf_url = mirror + pdf_url

                        print(f"   🎯 找到 PDF 真实地址: {pdf_url}")
                        
                        # 提取过了验证的 Cookie 给 requests 用
                        browser_cookies = context.cookies()
                        cookie_dict = {c['name']: c['value'] for c in browser_cookies}
                        headers = {"User-Agent": context.request.DEFAULT_OPTIONS.get("user_agent", "")}

                        print("   ⬇️  正在写入本地文件...")
                        # 使用 requests 进行真实的二进制下载 (防止浏览器直接预览)
                        r = requests.get(pdf_url, cookies=cookie_dict, headers=headers, verify=False, stream=True, timeout=60)
                        
                        if "application/pdf" in r.headers.get("Content-Type", "").lower() or r.content[:4] == b'%PDF':
                            with open(save_path, "wb") as f:
                                for chunk in r.iter_content(chunk_size=8192):
                                    f.write(chunk)
                            print(f"   ✅ [Success] 下载成功: {save_path}")
                            
                            browser.close()
                            return save_path
                        else:
                            print("   ⚠️ 下载的文件不是 PDF，可能是其他验证页面。")
                    else:
                        print("   ⚠️ 页面已加载，但未找到 PDF 链接标签。")

                except Exception as e:
                    print(f"   ⚠️ 镜像 {mirror} 处理出错: {e}")

            browser.close()
            
        print(f"❌ [Failed] 所有 Sci-Hub 镜像均无法下载此 DOI。")
        return None

# --- 测试用例 ---
# if __name__ == "__main__":
#     downloader = SciHubPlaywrightDownloader()
    
#     # 测试你刚才触发验证的这篇论文
#     test_doi = "10.1109/TCCN.2017.2758370"
#     downloader.download(test_doi)


import os
import requests
import re
import urllib3
from playwright.sync_api import sync_playwright

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class SciHubPlaywrightDownloader:
    def __init__(self, save_dir="./pdfs_scihub"):
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        # 把刚发现的有效域名 .sg 也加进去
        self.mirrors = [
            "https://sci-hub.st", # 既然这个能跳转，放前面
            "https://sci-hub.sg", 
            "https://sci-hub.se", 
            "https://sci-hub.ru"
        ]

    def download(self, doi):
        clean_doi = doi.replace("https://doi.org/", "").replace("http://dx.doi.org/", "")
        filename = f"{clean_doi.replace('/', '_')}.pdf"
        save_path = os.path.join(self.save_dir, filename)

        if os.path.exists(save_path):
            print(f"✅ [Exist] 文件已存在: {filename}")
            return save_path

        print(f"🔍 [Sci-Hub] 正在搜索 DOI: {clean_doi}")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
            )
            page = context.new_page()

            for mirror in self.mirrors:
                target_url = f"{mirror}/{clean_doi}"
                try:
                    print(f"   🌐 正在访问镜像: {target_url}")
                    
                    # ==========================================
                    # 核心修复：容忍重定向 (Redirect)
                    # ==========================================
                    try:
                        # wait_until="commit" 表示只要服务器响应了就行，不管是不是重定向
                        page.goto(target_url, timeout=30000, wait_until="commit")
                    except Exception as e:
                        if "interrupted" in str(e):
                            print("   🔄 捕获到网站重定向跳转，正在跟随...")
                        else:
                            raise e # 其他真实的网络错误（比如 ERR_NAME_NOT_RESOLVED）继续抛出

                    # 等待最终重定向后的页面加载完毕
                    try:
                        page.wait_for_load_state("domcontentloaded", timeout=20000)
                    except Exception:
                        pass # 有时即使超时，DOM也已经够用了，不强行中断

                    # 打印最终落地网址，确认是不是跳到了 .sg
                    print(f"   📍 最终落地网址: {page.url}")

                    # ==========================================
                    # 以下处理人机验证和提取 PDF 逻辑保持不变
                    # ==========================================
                    
                    # 1. 自动处理 "Are you a robot?" 验证
                    if page.locator(".answer").is_visible() or "robot" in page.content().lower():
                        print("   🤖 触发人机验证，正在自动点击 'No' 进行破解...")
                        try:
                            page.locator(".answer").click()
                            page.wait_for_navigation(timeout=15000)
                            print("   🔓 验证破解成功，页面已刷新！")
                        except Exception as e:
                            print(f"   ⚠️ 验证破解失败: {e}")
                            continue 

                    # 2. 如果页面显示 article not found
                    if "article not found" in page.content().lower():
                        print(f"   ❌ [Not Found] 此文章未被收录。")
                        break 

                    # 3. 提取真实的 PDF 地址
                    pdf_url = None
                    print(f"{page.content()}")
                    pdf_element = page.locator("iframe#pdf, embed#pdf").first
                    if pdf_element.count() > 0:
                        pdf_url = pdf_element.get_attribute("src")
                    
                    if not pdf_url:
                        html_content = page.content()
                        match = re.search(r"location\.href='([^']+)'", html_content)
                        if match:
                            pdf_url = match.group(1)

                    # 4. 桥接 Cookie 并下载 PDF
                    if pdf_url:
                        if pdf_url.startswith("//"):
                            pdf_url = "https:" + pdf_url
                        elif pdf_url.startswith("/"):
                            # 注意：这里要用最终落地的域名拼接，而不是初始 mirror
                            from urllib.parse import urlparse
                            parsed_url = urlparse(page.url)
                            base_domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
                            pdf_url = base_domain + pdf_url

                        print(f"   🎯 找到 PDF 真实地址: {pdf_url}")
                        
                        browser_cookies = context.cookies()
                        cookie_dict = {c['name']: c['value'] for c in browser_cookies}
                        headers = {"User-Agent": context.request.DEFAULT_OPTIONS.get("user_agent", "")}

                        print("   ⬇️  正在写入本地文件...")
                        r = requests.get(pdf_url, cookies=cookie_dict, headers=headers, verify=False, stream=True, timeout=60)
                        
                        if "application/pdf" in r.headers.get("Content-Type", "").lower() or r.content[:4] == b'%PDF':
                            with open(save_path, "wb") as f:
                                for chunk in r.iter_content(chunk_size=8192):
                                    f.write(chunk)
                            print(f"   🎉 [Success] 下载成功: {save_path}")
                            browser.close()
                            return save_path
                        else:
                            print("   ⚠️ 下载的文件不是 PDF，可能是其他验证页面。")
                    else:
                        print("   ⚠️ 页面已加载，但未找到 PDF 链接标签。")

                except Exception as e:
                    # 只有真正的网络错误会走到这里
                    if "ERR_NAME_NOT_RESOLVED" in str(e) or "ERR_CONNECTION_CLOSED" in str(e):
                        print(f"   ⚠️ 镜像 {mirror} 被网络阻断。")
                    else:
                        print(f"   ⚠️ 镜像 {mirror} 处理出错: {e}")

            browser.close()
            
        print(f"❌ [Failed] 所有 Sci-Hub 镜像均无法下载此 DOI。")
        return None

# # --- 测试用例 ---
# if __name__ == "__main__":
#     downloader = SciHubPlaywrightDownloader()
    
#     # 测试这篇论文
#     test_doi = "10.1109/TCCN.2017.2758370"
#     downloader.download(test_doi)


import os
import requests
import re
import urllib3
from playwright.sync_api import sync_playwright

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class SciHubPlaywrightDownloader:
    def __init__(self, save_dir="./pdfs_scihub"):
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        # 优先级排序：从最容易跳转的开始
        self.mirrors = [
            "https://sci-hub.st", 
            "https://sci-hub.sg", 
            "https://sci-hub.se", 
            "https://sci-hub.ru"
        ]

    def download(self, doi):
        clean_doi = doi.replace("https://doi.org/", "").replace("http://dx.doi.org/", "")
        filename = f"{clean_doi.replace('/', '_')}.pdf"
        save_path = os.path.join(self.save_dir, filename)

        if os.path.exists(save_path):
            print(f"✅ [Exist] 文件已存在: {filename}")
            return save_path

        print(f"🔍 [Sci-Hub] 正在搜索 DOI: {clean_doi}")

        with sync_playwright() as p:
            # 建议保持 headless=True，如果一直过不去验证，可以改成 False 看看具体情况
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                # 必须伪装完美的 User-Agent，否则 Altcha 会直接拒绝计算
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
                viewport={"width": 1280, "height": 800}
            )
            page = context.new_page()

            for mirror in self.mirrors:
                target_url = f"{mirror}/{clean_doi}"
                try:
                    print(f"   🌐 正在访问镜像: {target_url}")
                    
                    # 1. 容忍重定向
                    try:
                        page.goto(target_url, timeout=30000, wait_until="commit")
                    except Exception as e:
                        if "interrupted" in str(e):
                            print("   🔄 捕获到网站重定向跳转，正在跟随...")
                        else:
                            raise e 

                    # 等待 DOM 加载，不强求 networkidle
                    try:
                        page.wait_for_load_state("domcontentloaded", timeout=20000)
                        # 稍微等一下让 JS 渲染
                        page.wait_for_timeout(2000) 
                    except Exception:
                        pass 

                    print(f"   📍 当前网页地址: {page.url}")

                    # ==========================================
                    # 核心突破口：自动处理 Altcha 人机验证
                    # ==========================================
                    if "are you a robot" in page.content().lower() or page.locator(".answer").count() > 0:
                        print("   🤖 触发人机验证 (Altcha PoW Hash)，准备破解...")
                        try:
                            # 确保按钮存在且可见
                            page.wait_for_selector(".answer", state="visible", timeout=10000)
                            
                            # 停顿 2 秒，等待 altcha-widget 的 JS 库完全初始化
                            page.wait_for_timeout(2000)
                            
                            # 【绝杀】不要模拟点击，直接在控制台执行网页原生的 check() 函数！
                            print("   ⚙️ 正在向浏览器注入执行指令...")
                            page.evaluate("check()")
                            
                            print("   ⏳ 开始进行哈希计算，等待页面自动刷新 (请耐心等待 10~30 秒)...")
                            
                            # 【绝杀】不要等 navigation，直接死等 PDF 元素出现！
                            # 只要 PDF 框或者 Save 按钮出现了，就说明验证通过且页面刷新完了
                            page.wait_for_selector("iframe#pdf, embed#pdf, div#buttons", timeout=45000)
                            
                            print("   🔓 验证破解成功，已进入论文页面！")
                        except Exception as e:
                            print(f"   ⚠️ 破解失败或计算超时: {e}")
                            # 截图保存死因
                            page.screenshot(path="captcha_failed.png")
                            continue # 这个镜像废了，试下一个

                    # ==========================================
                    # 提取 PDF 和下载 (保持不变)
                    # ==========================================
                    if "article not found" in page.content().lower():
                        print(f"   ❌ [Not Found] 此文章未被收录。")
                        break 

                    pdf_url = None
                    pdf_element = page.locator("iframe#pdf, embed#pdf").first
                    if pdf_element.count() > 0:
                        pdf_url = pdf_element.get_attribute("src")
                    
                    if not pdf_url:
                        html_content = page.content()
                        match = re.search(r"location\.href='([^']+)'", html_content)
                        if match:
                            pdf_url = match.group(1)

                    if pdf_url:
                        if pdf_url.startswith("//"):
                            pdf_url = "https:" + pdf_url
                        elif pdf_url.startswith("/"):
                            from urllib.parse import urlparse
                            parsed_url = urlparse(page.url)
                            base_domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
                            pdf_url = base_domain + pdf_url

                        print(f"   🎯 找到 PDF 真实地址: {pdf_url}")
                        
                        browser_cookies = context.cookies()
                        cookie_dict = {c['name']: c['value'] for c in browser_cookies}
                        headers = {"User-Agent": context.request.DEFAULT_OPTIONS.get("user_agent", "")}

                        print("   ⬇️  正在写入本地文件...")
                        r = requests.get(pdf_url, cookies=cookie_dict, headers=headers, verify=False, stream=True, timeout=60)
                        
                        if "application/pdf" in r.headers.get("Content-Type", "").lower() or r.content[:4] == b'%PDF':
                            with open(save_path, "wb") as f:
                                for chunk in r.iter_content(chunk_size=8192):
                                    f.write(chunk)
                            print(f"   🎉 [Success] 下载成功: {save_path}")
                            browser.close()
                            return save_path
                        else:
                            print("   ⚠️ 下载的文件不是 PDF。")
                    else:
                        print("   ⚠️ 页面已加载，但未找到 PDF 链接标签。")
                        page.screenshot(path="no_pdf_link.png")

                except Exception as e:
                    if "ERR_NAME_NOT_RESOLVED" in str(e) or "ERR_CONNECTION_CLOSED" in str(e):
                        pass # 静默处理纯网络阻断
                    else:
                        print(f"   ⚠️ 镜像 {mirror} 处理出错: {e}")

            browser.close()
            
        print(f"❌ [Failed] 所有 Sci-Hub 镜像均无法下载此 DOI。")
        return None

# --- 测试用例 ---
if __name__ == "__main__":
    downloader = SciHubPlaywrightDownloader()
    test_doi = "10.1109/TCCN.2017.2758370"
    downloader.download(test_doi)