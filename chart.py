import matplotlib.pyplot as plt

# Data for the chart
sentiments = ['Negative', 'Neutral', 'Positive']
precision = [0.81, 0.00, 0.80]
recall = [1.00, 0.00, 0.17]
f1_score = [0.89, 0.00, 0.28]

# Create a bar chart
bar_width = 0.25
r1 = range(len(precision))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

plt.figure(figsize=(10, 7))

plt.bar(r1, precision, width=bar_width, label='Precision', color='blue')
plt.bar(r2, recall, width=bar_width, label='Recall', color='green')
plt.bar(r3, f1_score, width=bar_width, label='F1-Score', color='red')

# Adding labels to the chart
plt.xlabel('Sentiment', fontweight='bold')
plt.xticks([r + bar_width for r in range(len(precision))], sentiments)
plt.ylabel('Score', fontweight='bold')
plt.title('Performance Metrics by Sentiment', fontweight='bold')
plt.legend()

# Display the chart
plt.tight_layout()
plt.show()