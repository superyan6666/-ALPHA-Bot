import requests

def test():
    url = "http://qt.gtimg.cn/q=sh600519"
    resp = requests.get(url, timeout=5)
    resp.encoding = 'gbk'
    data = resp.text.strip().split('=', 1)[1]
    parts = data.replace('"', '').replace(';', '').split('~')
    for i, p in enumerate(parts):
        print(f"{i}: {p}")

if __name__ == "__main__":
    test()
