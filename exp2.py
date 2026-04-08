graph = {
	'A' : {'B' : 1,'C' :3},
	'B' : {'D' :2 },
	'C' : {'D' :1},
	'D' : {}

}


heursitic = { 
	'A' :3,
	'B' :2,
	'C' :1,
	'D' :0
}

def a_star(start, goal):
	open_list = [(start ,0)]
	
	while open_list:
		node, cost = open_list.pop(0)	

		print("Visiting:", node)
	
	if node == goal:
		print("Goal reached with Cost:",cost)
		return	
	
	for neighbor in graph[node]:
		total_cost = cost + graph[node][neighbor]
		open_list.append((neighbor, total_cost))

a_star('A','D')
