from DrissionPage import Chromium

# 启动或接管浏览器，并获取标签页对象
tab = Chromium().latest_tab
# 跳转到登录页面
tab.get('https://medcloud.sjtu.edu.cn/workplaceNV.php')

# 等待页面加载完成
tab.wait.doc_loaded()

# 添加调试信息，查看页面内容
print("=== 调试信息 ===")
print(f"当前页面标题: {tab.title}")
print(f"当前页面URL: {tab.url}")

# 等待更长时间确保页面完全加载
import time
time.sleep(3)

# 尝试查找所有tab-header-div类的元素
tab_header_elements = tab.eles('.tab-header-div')
print(f"tab-header-div类的元素数量: {len(tab_header_elements)}")

# 打印所有tab-header-div元素的详细信息
print("\n=== 所有tab-header-div元素的详细信息 ===")
for i, element in enumerate(tab_header_elements):
    print(f"\n元素 {i+1}:")
    print(f"  标签名: {element.tag}")
    print(f"  文本内容: {element.text}")
    print(f"  HTML: {element.html}")
    print(f"  属性: {element.attrs}")
    print("-" * 50)

# 检查第5个tab-header-div元素的文本，如果是"完成"就点击
try:
    # 获取所有tab-header-div元素
    tab_header_elements = tab.eles('.tab-header-div')
    
    # 检查是否有足够的元素
    if len(tab_header_elements) >= 5:
        # 获取第5个元素（索引为4）
        element_5 = tab_header_elements[4]
        element_text = element_5.text.strip()  # 获取文本并去除空白字符
        
        # 检查文本是否为"完成"
        if "完成" in element_text:
            element_5.click()
            print(f"第5个元素文本为'{element_text}'，包含'完成'，已成功点击")
        else:
            print(f"第5个元素文本为'{element_text}'，不包含'完成'，跳过点击")
    else:
        print(f"页面中只有{len(tab_header_elements)}个tab-header-div元素，无法检查第5个")
except Exception as e:
    print(f"检查第5个元素时出错: {e}")

# 定义处理单页的函数
def process_single_page(page_number):
    print(f"\n{'='*60}")
    print(f"开始处理第 {page_number} 页")
    print(f"{'='*60}")
    
    # 访问指定页面
    page_url = f'https://medcloud.sjtu.edu.cn/workplaceNV.php?pageno={page_number}'
    tab.get(page_url)
    tab.wait.doc_loaded()
    time.sleep(2)  # 等待页面加载
    
    # 查找指定的table元素，并点击其下属class="row"的每一个元素
    try:
        # 查找指定的table元素
        table_elements = tab.eles('@@class=table table-striped table-bordered table-hover dataTables-example@@id=idcode5')
        print(f"\n找到 {len(table_elements)} 个指定的table元素")
        
        if len(table_elements) > 0:
            # 遍历每个table元素
            for table_index, table_element in enumerate(table_elements):
                print(f"\n=== 处理第 {table_index + 1} 个table元素 ===")
                print(f"table元素id: {table_element.attr('id')}")
                
                # 在该table元素下查找所有class="row"的元素
                row_elements = table_element.eles('.row')
                print(f"该table下有 {len(row_elements)} 个class='row'的元素")
                
                if len(row_elements) > 0:
                    # 遍历每个row元素
                    for row_index, row_element in enumerate(row_elements):
                        print(f"  处理第 {row_index + 1} 个row元素...")
                        
                        # 在该row元素下查找所有tr元素（表格行）
                        tr_elements = row_element.eles('@tag()=tr')
                        print(f"    该row下有 {len(tr_elements)} 个tr元素（表格行）")
                        
                        if len(tr_elements) > 0:
                            # 遍历并点击每个tr元素（表格行）
                            for tr_index, tr_element in enumerate(tr_elements):
                                try:
                                    print(f"      正在点击第 {tr_index + 1} 个tr元素（表格行）...")
                                    print(f"        tr元素文本: {tr_element.text}")
                                    
                                    # 点击tr元素（表格行）
                                    tr_element.click()
                                    print(f"        成功点击第 {tr_index + 1} 个tr元素（表格行）")
                                    
                                    # 等待一下，让页面响应
                                    time.sleep(1)
                                    
                                    # 点击后定位唯一的class="col-sm-8"元素，并查找符合条件的下载链接
                                    try:
                                        col_sm_8_element = tab.ele('.col-sm-12')
                                        if col_sm_8_element:
                                            print(f"        第 {tr_index + 1} 个tr点击后，找到唯一的col-sm-8元素:")
                                            print(f"          col-sm-8元素HTML: {col_sm_8_element.html}")
                                            
                                            # 在col-sm-8元素中查找所有有style属性的元素
                                            style_elements = col_sm_8_element.eles('@tag()=tr')
                                            print(f"          找到 {len(style_elements)} 个tr元素")
                                            
                                            for style_index, style_element in enumerate(style_elements):
                                                try:
                                                    # 获取元素的文本内容
                                                    element_text = style_element.text.strip()
                                                    print(f"          元素 {style_index + 1} 文本: {element_text}")
                                                    
                                                    # 检查文本是否包含"stl"
                                                    if "stl" in element_text.lower():
                                                        print(f"          发现包含'stl'的元素: {element_text}")
                                                        
                                                        # 查找该元素中的href链接
                                                        href_elements = style_element.eles('@href')
                                                        if len(href_elements) > 0:
                                                            for href_index, href_element in enumerate(href_elements):
                                                                href_url = href_element.attr('href')
                                                                if href_url:
                                                                    print(f"            找到下载链接 {href_index + 1}: {href_url}")
                                                                    
                                                                    # 这里可以添加下载逻辑
                                                                    # 例如：download_file(href_url)
                                                                    print(f"            准备下载: {href_url}")
                                                                    
                                                                    # 点击下载链接
                                                                    try:
                                                                        href_element.click()
                                                                        print(f"            成功点击下载链接")
                                                                        time.sleep(2)  # 等待下载开始
                                                                    except Exception as download_e:
                                                                        print(f"            点击下载链接失败: {download_e}")
                                                                else:
                                                                    print(f"            元素 {href_index + 1} 没有有效的href属性")
                                                        else:
                                                            print(f"          该元素中没有找到href链接")
                                                    else:
                                                        print(f"          元素不包含'stl'，跳过")
                                                        
                                                except Exception as element_e:
                                                    print(f"          处理元素 {style_index + 1} 时出错: {element_e}")
                                            
                                        else:
                                            print(f"        第 {tr_index + 1} 个tr点击后，未找到col-sm-8元素")
                                            
                                    except Exception as col_e:
                                        print(f"        查找col-sm-8元素时出错: {col_e}")
                                    
                                    # 等待一下，避免点击过快
                                    time.sleep(0.5)
                                    
                                except Exception as e:
                                    print(f"        点击第 {tr_index + 1} 个tr元素时出错: {e}")
                        else:
                            print("    该row下没有找到tr元素（表格行）")
                else:
                    print("  该table下没有找到class='row'的元素")
        else:
            print("未找到指定的table元素")
            
    except Exception as e:
        print(f"查找table元素时出错: {e}")
    
    print(f"\n第 {page_number} 页处理完成")
    print(f"{'='*60}")

# 遍历第1到42页
for page_num in range(2, 43):  # range(1, 43) 表示1到42
    try:
        process_single_page(page_num)
        print(f"第 {page_num} 页处理完成，等待3秒后继续下一页...")
        time.sleep(3)  # 每页之间等待3秒
    except Exception as e:
        print(f"处理第 {page_num} 页时出错: {e}")
        print("继续处理下一页...")
        time.sleep(3)

# 等待点击后的页面响应
tab.wait.doc_loaded()
