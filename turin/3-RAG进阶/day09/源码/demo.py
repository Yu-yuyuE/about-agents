import requests


headers = {
    "Accept": "*/*",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "Origin": "http://www.cninfo.com.cn",
    "Pragma": "no-cache",
    "Referer": "http://www.cninfo.com.cn/new/commonUrl?url=disclosure/list/notice",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
    "X-Requested-With": "XMLHttpRequest"
}
cookies = {
    "JSESSIONID": "300BA28F05BE06BC1A8DAB3E87C43B60",
    "SF_cookie_4": "17470996",
    "_sp_ses.2141": "*",
    "_sp_id.2141": "e197c44c-a06f-41b7-98c1-f266c2b0ce12.1756988163.4.1761568474.1760276273.f8f0ac4b-b04f-4d15-88d6-ec424b81f33c",
    "insert_cookie": "37836164"
}
url = "http://www.cninfo.com.cn/new/disclosure"
data = {
    "column": "szse_latest",
    "pageNum": "6",
    "pageSize": "30",
    "sortName": "",
    "sortType": "",
    "clusterFlag": "true"
}
response = requests.post(url, headers=headers, cookies=cookies, data=data, verify=False)

print(response.text)
print(response)