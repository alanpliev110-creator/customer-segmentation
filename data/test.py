import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("data/onlinII.csv", encoding="ISO-8859-1")

print(df.head())
print(df.info())


# Drop missing Customer IDs
df = df.dropna(subset=['Customer ID'])

# Remove cancellations and adjustments
df = df[df['Quantity'] > 0]
df = df[df['Price'] > 0]

# Remove duplicates
df = df.drop_duplicates()

# Confirm clean shape
print(f"Cleaned dataset: {df.shape}")
print(f"Rows removed: {1067371 - df.shape[0]:,}")




# Create revenue column
df['Revenue'] = df['Quantity'] * df['Price']

# Convert date column to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Extract month and year for trend analysis
df['YearMonth'] = df['InvoiceDate'].dt.to_period('M')

print(df[['Quantity', 'Price', 'Revenue', 'InvoiceDate', 'YearMonth']].head(10))
print(f"\nDate range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")





monthly_revenue = df.groupby('YearMonth')['Revenue'].sum().reset_index()
monthly_revenue['YearMonth'] = monthly_revenue['YearMonth'].astype(str)

plt.figure(figsize=(14, 5))
sns.lineplot(data=monthly_revenue, x='YearMonth', y='Revenue', marker='o', color='steelblue')
plt.xticks(rotation=45, ha='right')
plt.title('Monthly Revenue — Online Retail II (2009–2011)', fontsize=14, fontweight='bold')
plt.xlabel('Month')
plt.ylabel('Revenue (£)')
plt.tight_layout()
plt.show()




country_revenue = df.groupby('Country')['Revenue'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(12, 5))
sns.barplot(x=country_revenue.index, y=country_revenue.values, color='steelblue')
plt.title('Top 10 Countries by Revenue', fontsize=14, fontweight='bold')
plt.xlabel('Country')
plt.ylabel('Revenue (£)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print(country_revenue)





country_revenue = df.groupby('Country')['Revenue'].sum().sort_values(ascending=False).head(10)
country_revenue_df = country_revenue.reset_index()
country_revenue_df.columns = ['Country', 'Revenue']
country_revenue_df['% of Total Revenue'] = (country_revenue_df['Revenue'] / df['Revenue'].sum() * 100).round(2)

print(country_revenue_df)





country_revenue_excl_uk = df[df['Country'] != 'United Kingdom'].groupby('Country')['Revenue'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(12, 5))
sns.barplot(x=country_revenue_excl_uk.index, y=country_revenue_excl_uk.values, color='steelblue')
plt.title('Top 10 Countries by Revenue (Excl. UK)', fontsize=14, fontweight='bold')
plt.xlabel('Country')
plt.ylabel('Revenue (£)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print(country_revenue_excl_uk)




# Filter out non-product entries
non_products = ['Manual', 'POSTAGE', 'DOTCOM POSTAGE', 'Discount']

product_revenue = df[~df['Description'].isin(non_products)].groupby('Description')['Revenue'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(12, 5))
sns.barplot(x=product_revenue.index, y=product_revenue.values, color='steelblue')
plt.title('Top 10 Products by Revenue', fontsize=14, fontweight='bold')
plt.xlabel('Product')
plt.ylabel('Revenue (£)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print(product_revenue)



monthly_orders = df.groupby('YearMonth')['Invoice'].nunique().reset_index()
monthly_orders.columns = ['YearMonth', 'Orders']
monthly_orders['YearMonth'] = monthly_orders['YearMonth'].astype(str)

plt.figure(figsize=(14, 5))
sns.lineplot(data=monthly_orders, x='YearMonth', y='Orders', marker='o', color='steelblue')
plt.xticks(rotation=45, ha='right')
plt.title('Monthly Order Volume — Online Retail II (2009–2011)', fontsize=14, fontweight='bold')
plt.xlabel('Month')
plt.ylabel('Number of Orders')
plt.tight_layout()
plt.show()




monthly_aov = df.groupby('YearMonth').apply(
    lambda x: x.groupby('Invoice')['Revenue'].sum().mean()
).reset_index()
monthly_aov.columns = ['YearMonth', 'AOV']
monthly_aov['YearMonth'] = monthly_aov['YearMonth'].astype(str)

plt.figure(figsize=(14, 5))
sns.lineplot(data=monthly_aov, x='YearMonth', y='AOV', marker='o', color='steelblue')
plt.xticks(rotation=45, ha='right')
plt.title('Average Order Value by Month — Online Retail II (2009–2011)', fontsize=14, fontweight='bold')
plt.xlabel('Month')
plt.ylabel('Average Order Value (£)')
plt.tight_layout()
plt.show()




# Get each customer's first ever purchase date
first_purchase = df.groupby('Customer ID')['InvoiceDate'].min().reset_index()
first_purchase.columns = ['Customer ID', 'FirstPurchaseDate']
first_purchase['FirstPurchaseMonth'] = first_purchase['FirstPurchaseDate'].dt.to_period('M')

# Merge back to main dataframe
df = df.merge(first_purchase[['Customer ID', 'FirstPurchaseMonth']], on='Customer ID', how='left')

# Tag each transaction as new or returning
df['CustomerType'] = df.apply(
    lambda x: 'New' if x['YearMonth'] == x['FirstPurchaseMonth'] else 'Returning', axis=1
)

# Count distinct customers per month per type
customer_type_monthly = df.groupby(['YearMonth', 'CustomerType'])['Customer ID'].nunique().reset_index()
customer_type_monthly.columns = ['YearMonth', 'CustomerType', 'Customers']
customer_type_monthly['YearMonth'] = customer_type_monthly['YearMonth'].astype(str)

plt.figure(figsize=(14, 5))
sns.lineplot(data=customer_type_monthly, x='YearMonth', y='Customers', hue='CustomerType', marker='o')
plt.xticks(rotation=45, ha='right')
plt.title('New vs Returning Customers by Month', fontsize=14, fontweight='bold')
plt.xlabel('Month')
plt.ylabel('Number of Customers')
plt.tight_layout()
plt.show()



import datetime as dt

# Set reference date as one day after the last transaction
reference_date = df['InvoiceDate'].max() + dt.timedelta(days=1)

# Build RFM table
rfm = df.groupby('Customer ID').agg(
    Recency=('InvoiceDate', lambda x: (reference_date - x.max()).days),
    Frequency=('Invoice', 'nunique'),
    Monetary=('Revenue', 'sum')
).reset_index()

print(rfm.shape)
print(rfm.head(10))




print(f"Total unique customers: {rfm.shape[0]}")
print(f"\nRecency (days):")
print(rfm['Recency'].describe())
print(f"\nFrequency (orders):")
print(rfm['Frequency'].describe())
print(f"\nMonetary (£):")
print(rfm['Monetary'].describe())




# Score Recency — inverted because lower days = better
rfm['R_Score'] = pd.qcut(rfm['Recency'], q=4, labels=[4, 3, 2, 1])

# Score Frequency
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=4, labels=[1, 2, 3, 4])

# Score Monetary
rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), q=4, labels=[1, 2, 3, 4])

# Combine into single RFM score
rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

print(rfm.head(10)) 



def assign_segment(row):
    r = int(row['R_Score'])
    f = int(row['F_Score'])
    m = int(row['M_Score'])
    
    if r >= 3 and f >= 3 and m >= 3:
        return 'Champion'
    elif r >= 3 and f >= 2:
        return 'Loyal'
    elif r >= 3 and f <= 2:
        return 'New/Promising'
    elif r == 2 and f >= 3:
        return 'At Risk'
    elif r == 2 and f <= 2:
        return 'Needs Attention'
    else:
        return 'Lost'

rfm['Segment'] = rfm.apply(assign_segment, axis=1)

print(rfm['Segment'].value_counts())



segment_order = ['Champion', 'Loyal', 'New/Promising', 'At Risk', 'Needs Attention', 'Lost']

segment_counts = rfm['Segment'].value_counts().reset_index()
segment_counts.columns = ['Segment', 'Count']

colors = {
    'Champion': '#2ecc71',
    'Loyal': '#3498db', 
    'New/Promising': '#9b59b6',
    'At Risk': '#e67e22',
    'Needs Attention': '#e74c3c',
    'Lost': '#95a5a6'
}

plt.figure(figsize=(10, 6))
sns.barplot(data=segment_counts, x='Segment', y='Count', 
            order=segment_order,
            palette=[colors[s] for s in segment_order])
plt.title('Customer Segments RFM Analysis', fontsize=14, fontweight='bold')
plt.xlabel('Segment')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()



segment_revenue = rfm.groupby('Segment')['Monetary'].sum().reset_index()
segment_revenue.columns = ['Segment', 'Revenue']
segment_revenue['% of Total Revenue'] = (segment_revenue['Revenue'] / segment_revenue['Revenue'].sum() * 100).round(2)

plt.figure(figsize=(10, 6))
sns.barplot(data=segment_revenue, x='Segment', y='Revenue',
            order=segment_order,
            palette=[colors[s] for s in segment_order])
plt.title('Revenue by Customer Segment', fontsize=14, fontweight='bold')
plt.xlabel('Segment')
plt.ylabel('Total Revenue (£)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print(segment_revenue.sort_values('Revenue', ascending=False))



rfm.to_csv('rfm_segments.csv', index=False)
print("RFM table exported successfully")
print(f"Shape: {rfm.shape}")
print(rfm.head())




df.to_csv('online_retail_cleaned.csv', index=False)
print("Cleaned dataset exported successfully")
print(f"Shape: {df.shape}")