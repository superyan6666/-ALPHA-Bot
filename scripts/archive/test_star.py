import baostock as bs
bs.login()
rs = bs.query_all_stock(day="2024-05-10")
count = 0
star_count = 0
while (rs.error_code == '0') and rs.next():
    count += 1
    code = rs.get_row_data()[0]
    if code.startswith('sh.688'):
        star_count += 1
print(f"Total: {count}, STAR: {star_count}")
bs.logout()
