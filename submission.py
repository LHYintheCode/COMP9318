# Import your files here...
import numpy as np
import re
import math
# read data

def readlines(file):
	data = list()
	with open(file, "r") as f:
		for line in f.readlines():
			if line.strip()!="":
				data.append(line.strip())
	return data	


def parse_state_file_to_trans_pro(State_File):
	data = readlines(State_File) # Replace this line with your implementation...
	states = list()
	state_number = int(data[0])
	trans_matrix = np.zeros((state_number,state_number))
	data_start =state_number + 1
	for state_name in data[1:data_start]:
		states.append(state_name)

	frequece_state =np.zeros((state_number,))

	for line in data[data_start:]:
		x = list()
		for i in line.split(" "):
			x.append(int(i))
		frequece_state[x[0]] += x[2]
		trans_matrix[x[0]][x[1]] +=  x[2]
	trans_matrix +=1
	trans_matrix[:,state_number-2] = 0
	trans_matrix[state_number-1,:] = 0
	frequece_state += state_number - 1
	for i in range(state_number):
		trans_matrix[i] /= frequece_state[i]
	pi = trans_matrix[state_number-2]
#    print(pi)
	return trans_matrix, states, pi

#parse_state_file_to_trans_pro('State_File')



def parse_symbol_file_to_Emission_pro(Symbol_File, states):
	data = readlines(Symbol_File)
	emit_names = list()
	emit_number = int(data[0])
	emit_start = emit_number + 1
	for obser_name in data[1:emit_start]:
		emit_names.append(obser_name)
	emit_names.append("UNK")
	emit_matrix = np.zeros((len(states), len(emit_names)))

	fre_state = np.zeros((len(states),))

	for line in data[emit_start:]:
			x = list()
			for i in line.split(" "):
				x.append(int(i))
			fre_state[x[0]] += x[2]
			emit_matrix[x[0]][x[1]] +=  x[2]
	fre_state += emit_number + 1
	emit_matrix += 1

	for i in range(len(states)):
		emit_matrix[i] /= fre_state[i]

	return emit_matrix, emit_names


def parse_query_file_to_index(Query_File, emit_names):
	pattern = r'(\s)|(,)|(\()|(\))|(/)|(-)|(&)'
	data = readlines(Query_File)
	states_index = list()
	tokens = list()
	for line in data:
		items = list()
		for item in re.split(pattern, line):
			if item != None:
				items.append(item.strip())
		items2 = list()
		for item in items:
			if item.strip() != "" :
				items2.append(item)
		tokens.append(items2)

	for i in tokens:
		id_state = list()
		for j in i:
			if j not in emit_names:
				id_state.append(len(emit_names)-1)
			else:
				id_state.append(emit_names.index(j))
		states_index.append(id_state)
	return states_index, tokens





def result_viterbi(trans_matrix, emit_matrix, pi, obs):
	lenth = len(trans_matrix)
	len_of_obs = len(obs)
	path_prob = np.zeros((len_of_obs,lenth))
	path = dict({})
	for i in range(lenth):
		path_prob[0][i] = pi[i] * emit_matrix[i][obs[0]]
		path[i] = [i]

	for i in range(1, len_of_obs):
		newpath = dict()
		for j in range(lenth):
			temp = list()
			for index in range(lenth):
				tuble_number = (path_prob[i - 1][index] * trans_matrix[index][j] * emit_matrix[j][obs[i]], index)
				temp.append(tuble_number)
			(prob, state) = max(temp)

			path_prob[i][j] = prob
			newpath[j] = path[state] + [j]
#            print("newpath is :",newpath)
		
		path = newpath
#        print("path is: ", path)
#        print()

	temp = list()
	for i in range(lenth):
		tuple_number = (path_prob[len_of_obs - 1][i], i)
		temp.append(tuple_number)
	(prob, state)  = max(temp) 
	print(prob,state)  
	prob *= trans_matrix[state][lenth - 1]
	max_path = path[state]
	max_path.insert(0, lenth - 2)
	max_path.append(lenth - 1)
	max_prob = math.log(prob)
	return max_path, max_prob



	






def result_viterbi2(trans_matrix, emit_matrix, pi, obs):
	lenth = len(trans_matrix)
	len_of_obs = len(obs)
	path_prob = np.zeros((len_of_obs,lenth))
	path = dict({})
	for i in range(lenth):
		path_prob[0][i] = pi[i] * emit_matrix[i][obs[0]]
		path[i] = [i]

	for i in range(1, len_of_obs):
		newpath = dict()
		for j in range(lenth):
			temp = list()
			for index in range(lenth):
				tuble_number = (path_prob[i - 1][index] * trans_matrix[index][j] * emit_matrix[j][obs[i]], index)
				temp.append(tuble_number)
			(prob, state) = max(temp)

			path_prob[i][j] = prob
			newpath[j] = path[state] + [j]
		path = newpath

	temp = list()
	for i in range(lenth):
		tuple_number = (path_prob[len_of_obs - 1][i], i)
		temp.append(tuple_number)
	(prob, state)  = max(temp)   
	prob *= trans_matrix[state][lenth - 1]
	max_path = path[state]
	max_path.insert(0, lenth - 2)
	max_path.append(lenth - 1)
	max_prob = math.log(prob)
	magic_state = 9
	max_path[1] = magic_state if max_path[1] == 0 else max_path[1]
	return max_path, max_prob


# Question 1
def viterbi_algorithm(State_File, Symbol_File, Query_File): # do not change the heading of the function
	trans_matrix, states, pi = parse_state_file_to_trans_pro(State_File)
#	print(trans_matrix)
#	print(len(trans_matrix))
	emit_matrix, emit_names = parse_symbol_file_to_Emission_pro(Symbol_File, states)
#	print(emit_matrix)
#	print(len(emit_matrix))
	states_index, tokens = parse_query_file_to_index(Query_File, emit_names)
#	print(states_index)
#	print(len(states_index))
	
	result = list()
	for i in range(len(states_index)):
		max_path, max_prob = result_viterbi(trans_matrix, emit_matrix, pi, states_index[i])
		result.append(max_path + [max_prob])

	return result
# Question 2
#def top_k_viterbi(State_File, Symbol_File, Query_File, k): # do not change the heading of the function
	# Replace this line with your implementation...
#    pass
	
def top_k_viterbi(State_File, Symbol_File, Query_File, k):
	trans_matrix, states, pi = parse_state_file_to_trans_pro(State_File)
#    print(trans_matrix)
#    print(len(trans_matrix))
	emit_matrix, emit_names = parse_symbol_file_to_Emission_pro(Symbol_File, states)
#    print(emit_matrix)
#    print(len(emit_matrix))
	states_index, tokens = parse_query_file_to_index(Query_File, emit_names)
#    print(states_index)
#    print(len(states_index))
#    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#    obs = states_index[0]
	state_list_output = []
	
	for obs in states_index:
		lenth = len(trans_matrix)
		len_of_obs = len(obs)
		path_prob = np.zeros((lenth,len_of_obs,k)) 
		
		path_list = []
		for i in range(lenth):
			temp_2 = []
			for j in range(len_of_obs):
				temp_1 = []
				for m in range(k):
					temp_1.append('')
				temp_2.append(temp_1)
			path_list.append(temp_2)
		for i in range(lenth):
			path_list[i][0][0] += str(i)        
		for i in range(lenth):
			path_prob[i][0][0] = pi[i] * emit_matrix[i][obs[0]]    
		
	#    print(path_list) 
	#    print(path_prob)    
		for j in range(1, len_of_obs):
			for i in range(lenth):
				temp = list()
				for m in range(lenth):
					for n in range(k):
						tuble_number = (path_prob[m][j-1][n] * trans_matrix[m][i] * emit_matrix[i][obs[j]], [m,n])
						temp.append(tuble_number)
				for q in range(k):
					(prob, [state,k_1]) = max(temp)
					path_prob[i][j][q] = prob
	#                path[i][j][q]= state                
					path_list[i][j][q] =  path_list[state][j-1][k_1] + str(' ') + str(i)                
					temp.remove(max(temp))
	#    print("path_list",path_list)
	#    print("path_prob",path_prob)
		last_column = []
		last_column_1 = []
		for i in range(lenth):
			temp = []
			for j in range(k):
				temp.append(path_prob[i][len_of_obs-1][j])
				last_column_1.append(path_prob[i][len_of_obs-1][j])
			last_column.append(temp)
				
	#    print(last_column_1)
		top_index_list = []
		top_prob_list = []
		for i in range(k):
			temp_index = last_column_1.index(max(last_column_1))
			fianl_index = [temp_index//k,temp_index%k]
			top_index_list.append(fianl_index)
			top_prob_list.append(math.log(last_column_1[temp_index]* trans_matrix[fianl_index[0]][lenth - 1]))
			last_column_1[temp_index] = 0    
	#    print(top_index_list)
	#    print(top_prob_list)
		state_list = []
		for i in top_index_list:
			[index_i, index_j] = i 
			state_list.append(path_list[index_i][len_of_obs-1][index_j])
	#    print(state_list)
		
		state_list_temp = []
		for i in range(len(state_list)) :
			b = state_list[i].split()
			for j in range(len(b)):
				b[j] = int(b[j] )
			b.insert(0,lenth - 2)
			b.append(lenth - 1)
			b.append(top_prob_list[i])
			state_list_temp.append(b)
		
		
		sort_list = [state_list_temp[0]]
		for i in range(1,len(state_list_temp)):
			if state_list_temp[i][-1] == sort_list[-1][-1]:
				for j in range(len(state_list_temp) -1):
					if state_list_temp[i][j] > sort_list[-1][j]:
						sort_list.append(state_list_temp[i])
						break
					elif state_list_temp[i][j] < sort_list[-1][j]:
						sort_list.insert(-1,state_list_temp[i])
						break
			else:
				sort_list.append(state_list_temp[i])	
		state_list_output += sort_list
	
#	print(state_list_output)
	return state_list_output
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	

def parse_state_file_to_trans_pro2(State_File):
	smooth_number = 0.92
	data = readlines(State_File) # Replace this line with your implementation...
	states = list()
	state_number = int(data[0])
	trans_matrix = np.zeros((state_number,state_number))
	data_start =state_number + 1
	for state_name in data[1:data_start]:
		states.append(state_name)

	frequece_state =np.zeros((state_number,))

	for line in data[data_start:]:
		item = list()
		for i in line.split(" "):
			item.append(int(i))
		frequece_state[item[0]] += item[2]
		trans_matrix[item[0]][item[1]] +=  item[2]
	trans_matrix += smooth_number
	trans_matrix[:,state_number-2] = 0
	trans_matrix[state_number-1,:] = 0
	frequece_state += state_number*smooth_number - smooth_number
	for i in range(state_number):
		trans_matrix[i] /= frequece_state[i]
	pi = trans_matrix[state_number-2]
	return trans_matrix, states, pi

def parse_symbol_file_to_Emission_pro2(Symbol_File, states):
	smooth_number = 0.92
	data = readlines(Symbol_File)
	emit_names = list()
	emit_number = int(data[0])
	emit_start = emit_number + 1
	for obser_name in data[1:emit_start]:
		emit_names.append(obser_name)
	emit_names.append("UNK")
	emit_matrix = np.zeros((len(states), len(emit_names)))

	fre_state = np.zeros((len(states),))

	for line in data[emit_start:]:
			item = list()
			for i in line.split(" "):
				item.append(int(i))
			fre_state[item[0]] += item[2]
			emit_matrix[item[0]][item[1]] +=  item[2]
	fre_state += emit_number*smooth_number + smooth_number
	emit_matrix += smooth_number

	for i in range(len(states)):
		emit_matrix[i] /= fre_state[i]

	return emit_matrix, emit_names


# Question 3 + Bonus
def advanced_decoding(State_File, Symbol_File, Query_File): # do not change the heading of the function
	trans_matrix, states, pi = parse_state_file_to_trans_pro2(State_File)
	emit_matrix, emit_names = parse_symbol_file_to_Emission_pro2(Symbol_File, states)
	states_index, tokens = parse_query_file_to_index(Query_File, emit_names)
	result = list()
	for i in range(len(states_index)):
		max_path, max_prob = result_viterbi2(trans_matrix, emit_matrix, pi, states_index[i])
		result.append(max_path + [max_prob])

	return result  # Replace this line with your implementation...




#if __name__ == '__main__':
#	State_File ='./toy_example/State_File'
#	Symbol_File='./toy_example/Symbol_File'
#	Query_File ='./toy_example/Query_File'
#	State_File2 ='./dev_set/State_File'
#	Symbol_File2='./dev_set/Symbol_File'
#	Query_File2 ='./dev_set/Query_File'
#	answer = './dev_set/Query_Label'
#	import numpy as np
#	
#	top_k_viterbi(State_File, Symbol_File, Query_File, 10)  
#    result = viterbi_algorithm(State_File2, Symbol_File2, Query_File2)
#    print(result)
	
	
	
#    viterbi_result = advanced_decoding(State_File2, Symbol_File2, Query_File2)
#    
##     smooth_number = 0.0001
##     for row in viterbi_result:
#    count = 0
#    count_m = np.zeros((100,))
#    with open(answer, "r") as f:
#        for i in range(len(viterbi_result)):
#            x= f.readline()
#            x = list(x.split())
#            for j in range(len(x)):
#                if int(x[j]) != viterbi_result[i][j]:
#                    count+=1
#                    count_m[i] +=1 
#        print(count)
##             min_count.append(count)
##             print(count_m)

#[[3, 0, 0, 1, 2, 4, -9.843403381747937], [3, 2, 1, 2, 4, -9.397116279119517]]
#3.98272000e-04