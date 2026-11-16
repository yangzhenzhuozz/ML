import re


def process(s: str) -> str:
    return re.sub(r"[^a-zA-Z]+", " ", s).lower()


with open("data.txt", encoding="utf-8") as file:
    result = process(file.read())

total_len = len(result)
if total_len == 0:
    print("文件内容为空，无法统计。")
else:
    statistics = [0] * 27
    for ch in result:
        idx = 26 if ch == " " else ord(ch) - ord("a")
        statistics[idx] += 1

    headers = [chr(i + ord("a")) if i != 26 else "space" for i in range(27)]

    # 打印表头
    print("| 字符 | 数量 | 占比 |")
    print("| :--- | :--- | :--- |")

    # 逐行打印每个字符的统计结果
    for char, count in sorted(
        zip(headers, statistics), key=lambda x: x[1], reverse=True
    ):
        percent = f"{(count / total_len * 100):.2f}%"
        print(f"| {char} | {count} | {percent} |")
