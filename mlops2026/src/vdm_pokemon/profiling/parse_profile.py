import pstats

#load stats file
stats = pstats.Stats('profile.stats')
stats.strip_dirs()

#save to txtfile
with open('profile_summary.txt', 'w') as f:
    stats.stream = f
    stats.sort_stats('cumulative').print_stats(20)

print("Summary saved to profile_summary.txt")
