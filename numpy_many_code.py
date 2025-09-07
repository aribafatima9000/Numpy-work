import numpy as np

# =======================
# Step 1: Create Sales Data (Products x Months)
# =======================
# 5 Products, 12 Months
sales = np.array([
    [120, 150, 100, 130, 160, 200, 210, 190, 180, 220, 250, 300],  # Product A
    [80, 100, 90, 95, 110, 120, 130, 125, 150, 170, 200, 220],    # Product B
    [200, 220, 250, 270, 300, 310, 330, 340, 360, 400, 420, 450], # Product C
    [60, 70, 65, 75, 80, 85, 95, 100, 110, 120, 130, 140],        # Product D
    [150, 180, 170, 160, 200, 220, 210, 230, 240, 260, 280, 300]  # Product E
])

products = ["A", "B", "C", "D", "E"]

print("📌 Sales Data (Products x Months):")
print(sales)

# =======================
# Step 2: Total Yearly Sales per Product
# =======================
total_sales = np.sum(sales, axis=1)
print("\n📊 Total Sales per Product:")
for p, t in zip(products, total_sales):
    print(f"Product {p}: {t}")

# =======================
# Step 3: Average Sales per Month
# =======================
avg_monthly_sales = np.mean(sales, axis=0)
print("\n📊 Average Sales per Month:")
print(avg_monthly_sales)

# =======================
# Step 4: Best-Selling Product
# =======================
best_index = np.argmax(total_sales)
print(f"\n🏆 Best-Selling Product: {products[best_index]} with {total_sales[best_index]} sales")

# =======================
# Step 5: Threshold Analysis
# =======================
threshold = 2000
high_sellers = np.where(total_sales >= threshold, "High Seller", "Low Seller")
print("\n📊 High/Low Seller Classification:")
for p, status in zip(products, high_sellers):
    print(f"Product {p}: {status}")
