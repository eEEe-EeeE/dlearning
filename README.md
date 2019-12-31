阿里云 http://mirrors.aliyun.com/pypi/simple/  
中国科技大学 https://pypi.mirrors.ustc.edu.cn/simple/  
豆瓣(douban) http://pypi.douban.com/simple/  
清华大学 https://pypi.tuna.tsinghua.edu.cn/simple/   
中国科学技术大学 http://pypi.mirrors.ustc.edu.cn/simple/  


pip freeze > requirements.txt
使用notepad++等编辑器正则替换==.*为空
pip install -r requirements.txt --upgrade