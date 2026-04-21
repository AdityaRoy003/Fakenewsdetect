from ddgs import DDGS

print("Testing text search...")
with DDGS() as ddgs:
    res = list(ddgs.text("Aliens have just landed their motherships right on top of the Eiffel Tower!", max_results=3))
    print(res)

print("\nTesting news search...")
with DDGS() as ddgs:
    res2 = list(ddgs.news("Aliens have just landed their motherships right on top of the Eiffel Tower!", max_results=3))
    print(res2)
