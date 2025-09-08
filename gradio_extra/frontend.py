css="""
    /* 输入框、下拉框统一美化 */
    input, select, textarea {
        border: 1px solid #ddd !important;
        border-radius: 10px !important;
        padding: 10px 14px !important;
        font-size: 15px !important;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }
    
    /* 聚焦时高亮边框 */
    input:focus, select:focus, textarea:focus {
        border: 1px solid #6a11cb !important;
        box-shadow: 0 0 6px rgba(106,17,203,0.3);
        outline: none !important;
    }
    
    /* Label 统一样式 
    label {
        font-weight: 600 !important;
        color: #333 !important;
        margin-bottom: 6px !important;
        display: block !important;
    }*/
    
    /* 表格样式 */
    .gradio-dataframe table {
        border-collapse: collapse;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        font-size: 14px;
    }
    
    /* 表格表头 */
    .gradio-dataframe thead {
        background: linear-gradient(135deg, #6a11cb, #2575fc);
        color: white;
        font-weight: bold;
    }
    
    /* 表格单元格 */
    .gradio-dataframe td, 
    .gradio-dataframe th {
        padding: 10px 14px;
        border: 1px solid #eee;
    }
    
    /* 表格 hover 效果 */
    .gradio-dataframe tbody tr:hover {
        background-color: #f9f9ff;
    }


    .fancy-btn {  background: linear-gradient(135deg, #6a11cb, #2575fc); color: white !important;
    font-weight: bold;
    font-size: 16px;
    padding: 12px 24px;
    border-radius: 12px;
    border: none;
    cursor: pointer;
    transition: all 0.3s ease-in-out;
    box-shadow: 0 4px 10px rgba(0,0,0,0.2);}
    
    .fancy-btn:hover {
    background: linear-gradient(135deg, #2575fc, #6a11cb);
    transform: translateY(-3px);
    box-shadow: 0 6px 15px rgba(0,0,0,0.3);
    }
    
    .fancy-btn:active {
        transform: scale(0.95);
    }
    
    .green-btn {
       background: linear-gradient(135deg, #2e7d32, #66bb6a);
    }
    .selection-button {
        display:none;
    }
    .cell-menu-button {
       display:none;
    }
    .cell-selected{
      --ring-color: none;
    }
    .add-row-button {
       display:none;
    }
    /* 先隐藏所有按钮 */
    .cell-menu button {
      display: none;
    }
    
    /* 只保留第 3 个 */
    .cell-menu button:nth-child(3) {
      display: inline-block;  /* 或者 flex，看原来的布局 */
    }
    
    .red-btn {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
    }
    """

head="""
    <script>
    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            mutation.addedNodes.forEach((node) => {
                if (node.tagName === "TEXTAREA") {
                      // 找到它所在的 td
                    const td = node.closest("td");
                    // 找到所在的 tr
                    const tr = node.closest("tr");
                    if (!td || !tr) return;
    
                    // 获取所有列
                    const tds = Array.from(tr.children);
                    // 如果不是最后一列，则只读
                    if (td !== tds[tds.length - 1]) {
                        node.readOnly = true;
                    } else {
                        node.readOnly = false; // 最后一列可编辑
                    }
                }
                
            });
        });
    });
    observer.observe(document.body, { childList: true, subtree: true });
    </script>
    """