import re


def process(s: str) -> str:
    return re.sub(r"[^a-zA-Z]+", " ", s).lower()


result = ""
with open("data.txt") as file:
    file_str = file.read()
    result = process(file_str)

total_len = len(result)
space_num = len([s for s in result if s == " "])
print(
    f"字符总数:{total_len}，空格数量:{space_num},空格占比{(space_num/total_len*100):.2f}%"
)
