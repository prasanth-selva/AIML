graph = {
    'A': ['B', 'C'],
    'B': ['D','E'],
    'C': ['F', 'G'],
    'D': [],
    'E': [],
    'F': [],
    'G': []
    }

def bfs(start):
	visited = []
	queue = [start]
	
	while queue:
		node = queue.pop(0)
		if node not in visited:
			print(node, end =" ")
			visited.append(node)
							
			for neighbour in graph[node]:
				queue.append(neighbour)
print("BFS\n")
bfs('A')


def dfs(node, visited = None):
	if visited is None:
		visited = []

	if node not in visited:
		print(node, end =" ")
		visited.append(node)
		

		for neighbor in graph[node]:
			dfs(neighbor,visited)  


print("DFS\n")
dfs('A')
