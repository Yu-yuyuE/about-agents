



import requests


headers = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "Pragma": "no-cache",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
    "sec-ch-ua": "\"Google Chrome\";v=\"141\", \"Not?A_Brand\";v=\"8\", \"Chromium\";v=\"141\"",
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "\"Windows\""
}
cookies = {
    "zhishiTopicRequestTime": "1761745354412",
    "PSTM": "1760964266",
    "BAIDUID": "CDADB0D59AC42CF2E480DD38F8F0A8A2:FG=1",
    "BIDUPSID": "BB3C8281FAB5D9BF613466F32412F0C4",
    "BAIDUID_BFESS": "CDADB0D59AC42CF2E480DD38F8F0A8A2:FG=1",
    "ZFY": "3vUqyGhXIfCOJU5wvX2ZZ7CU88lGG8DmvmQS:A9cdcKw:C",
    "H_PS_PSSID": "63148_64982_65247_65311_65361_65539_65618_65723_65759_65776_65803_65838_65858_65915_65921_65925_65940_65941_65962_65966_65989",
    "H_WISE_SIDS": "63148_64982_65247_65311_65361_65539_65618_65723_65759_65776_65803_65838_65858_65915_65921_65925_65940_65941_65962_65966_65989",
    "ab_sr": "1.0.1_NzlhZTU5ZDA3YjdiN2RkMzMwN2I2NjVjNTNjNzJkNjk4NDA5NWY3ZGZlN2I3NjhhZjUyZTFhOTdjYTZmZjRkYWE2NDhmNzBkZjA1OGU4ZWVjOGU3ODVlMjRjZWM0OWM5ZGZhMGRmODg3ZTMzZWM3N2M1NTcyZDcwYzQ2OWFkMjkwMzgyMjI4NTIxMjEzNWU3Nzc1MWE5YzIxNjAyOTJiOA==",
    "baikeVisitId": "13bee4af-f600-4bc9-a6c5-87d5720cee4a"
}
url = "https://baike.baidu.com/item/%E6%81%90%E9%BE%99/139019"
response = requests.get(url, headers=headers, cookies=cookies)

print(response.text)
print(response)