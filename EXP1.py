graph = {
    'A': ['B', 'C'],
    'B': ['D','E'],
    'C': ['F', 'G'],
    'D': [],
    'E': [],
    'F': [],
    }

def bfs(start):
    visited = []
    queue = [start]

    while queue:
        node = queue.pop(0)

        if node not in visited:
            print(node, end =" ")
            visited.append(node)
        
            for neighbor in graph[node]:
                queue.append(neighbor)
bfs('A')