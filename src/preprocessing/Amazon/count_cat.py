import gzip
import json
count = 0
cat_count = 0
with gzip.open("data/raw/amazon/book/meta_Books.json.gz") as f_in:
    for line_count, line in enumerate(f_in):
            product_data = json.loads(line.rstrip())
            count += 1
            if "category" in product_data:
                    cat_count += 1
print(count)
print(cat_count)
