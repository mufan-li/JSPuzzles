'''
	June2017.py

	Script to help solve the June 2017 'Well well well' puzzle.
'''

import numpy as np
import numpy.random as rd

class well(object):
	'''
		class well(object)

		defining the class of a well with all the related parameters
	'''
	def __init__(self, size = 3, seed = None, init_depth = None):
		super(well, self).__init__()
		self.size = size

		if init_depth == None:
			# set seed to repeat experiment
			if seed != None:
				rd.seed(seed)

			self.init_depth = np.arange(size ** 2).reshape((size,size)) + 1
			while self.init_depth[-1, -1] == size ** 2 or \
					self.init_depth[-1, -1] < (size-3) ** 2:
				rd.shuffle(self.init_depth)
				for row in self.init_depth:
					rd.shuffle(row)

			top_left = self.init_depth[0, 0]
			self.init_depth[self.init_depth == 1] = top_left
			self.init_depth[0, 0] = 1
		else:
			self.init_depth = init_depth

		self.current_depth = self.init_depth.copy()
		self.time_passed = 0.
		self.iter_time = 0.

		# flow convention 
		#	0 - no flow, i.e. equal
		#	1 - horizontal to the right, or vertically downwards
		#	-1 - horizontal to the left, or vertically upwards
		self.horz_flow_dir = np.zeros((size, size - 1))
		self.vert_flow_dir = np.zeros((size - 1, size))

		self.water_rate = np.zeros((size,size))

	def find_flow_index(self, row_from, col_from, row_to, col_to):
		'''
			def find_flow_index(self, row_from, col_from, row_to, col_to)

			wraps index finder in a function to easy calls
		'''
		if row_from == row_to:
			# horizontal flow 
			row_flow = row_from
			col_flow = min(col_from, col_to)
		else:
			# vertical flow
			row_flow = min(row_from, row_to)
			col_flow = col_from

		return row_flow, col_flow

	def update_flow_dir(self):
		'''
			def update_flow_dir(self)

			updates horz_flow_dir and vert_flow_dir
		'''

		# update horizontal flow direction
		for i in range(self.size):
			for j in range(self.size - 1):
				row_flow, col_flow = self.find_flow_index(i, j, i, j+1)
				if self.current_depth[i, j] > self.current_depth[i, j+1]:
					flow_dir = -1
				elif self.current_depth[i, j] < self.current_depth[i, j+1]:
					flow_dir = 1
				else:
					flow_dir = 0
				self.horz_flow_dir[row_flow, col_flow] = flow_dir

		# update vertical flow direction
		for i in range(self.size - 1):
			for j in range(self.size):
				row_flow, col_flow = self.find_flow_index(i, j, i+1, j)
				if self.current_depth[i, j] > self.current_depth[i+1, j]:
					flow_dir = -1
				elif self.current_depth[i, j] < self.current_depth[i+1, j]:
					flow_dir = 1
				else:
					flow_dir = 0
				self.vert_flow_dir[row_flow, col_flow] = flow_dir

	def check_flow_dir(self, row_from, col_from, row_to, col_to):
		'''
			def out_flow(self, row_from, col_from, row_to, col_to)

			determines whether flow in the direction of from-to
		'''
		# return no flow from out of bounds
		if min(row_from, col_from, row_to, col_to) < 0 or \
			max(row_from, col_from, row_to, col_to) >= self.size:
			return -1

		row_flow, col_flow = self.find_flow_index(row_from, col_from, row_to, col_to)
		# horizontal flow
		if row_from == row_to:
			flow_dir = self.horz_flow_dir[row_flow, col_flow]
			# left to right is positive
			if col_from < col_to:
				return flow_dir
			else:
				return -flow_dir
		# vertical flow
		else:
			flow_dir = self.vert_flow_dir[row_flow, col_flow]
			# up to down is positive
			if row_from < row_to:
				return flow_dir
			else:
				return -flow_dir


	def no_out_flow(self, row, col, count = False):
		'''
			def no_out_flow(self, row, col)

			determines whether the current index has any outflow 
		'''

		# equality indicates flow direction is equal
		up_flow = self.check_flow_dir(row, col, row - 1, col) > 0
		down_flow = self.check_flow_dir(row, col, row + 1, col) > 0
		left_flow = self.check_flow_dir(row, col, row, col - 1) > 0
		right_flow = self.check_flow_dir(row, col, row, col + 1) > 0

		if count:
			return int(up_flow) + int(down_flow) + int(left_flow) + int(right_flow)

		else:
			if up_flow or down_flow or left_flow or right_flow:
				return False
			else:
				return True


	def update_water_rate(self, rate = 1., row = 0, col = 0):
		'''
			def update_water_rate(self, rate = 0, row = 0, col = 0)

			recursive function to update the water rising rate at each cell

			note - the directional flow graph is none circular

			inputs
				rate - double, the additional rate that will be added to this cell
		'''
		# end recursion
		if self.no_out_flow(row, col):
			self.water_rate[row, col] += rate
			return

		up_flow = int(self.check_flow_dir(row, col, row - 1, col) > 0)
		down_flow = int(self.check_flow_dir(row, col, row + 1, col) > 0)
		left_flow = int(self.check_flow_dir(row, col, row, col - 1) > 0)
		right_flow = int(self.check_flow_dir(row, col, row, col + 1) > 0)

		next_rate = rate / (up_flow + down_flow + left_flow + right_flow)

		if up_flow:
			self.update_water_rate(next_rate, row - 1, col)
		if down_flow:
			self.update_water_rate(next_rate, row + 1, col)
		if left_flow:
			self.update_water_rate(next_rate, row, col - 1)
		if right_flow:
			self.update_water_rate(next_rate, row, col + 1)
		return

	def update_equal_levels(self, mask):
		'''
			def update_equal_levels(self)

			for neighbouring cells with equal depth, need to update the rate 
			such that they are equal
		'''
		each_rate = np.sum(self.water_rate[mask]) / np.sum(mask)
		self.water_rate[mask] = each_rate

	def check_neighbour_recursive(self, neighbour_mask, equal_depth_mask, row, col):
		'''
			def check_neighbour_recursive(self, neighbour_mask, equal_depth_mask, row, col)

			recursively find all neighbours in equal_depth_mask

			note will only recursively check if there is an unmarked neighbour
		'''

		if row - 1 >= 0 and not neighbour_mask[row - 1, col] and equal_depth_mask[row - 1, col]:
			neighbour_mask[row - 1, col] = True
			self.check_neighbour_recursive(neighbour_mask, equal_depth_mask, row - 1, col)

		if row + 1 < self.size and not neighbour_mask[row + 1, col] and equal_depth_mask[row + 1, col]:
			neighbour_mask[row + 1, col] = True
			self.check_neighbour_recursive(neighbour_mask, equal_depth_mask, row + 1, col)

		if col - 1 >= 0 and not neighbour_mask[row, col - 1] and equal_depth_mask[row, col - 1]:
			neighbour_mask[row, col - 1] = True
			self.check_neighbour_recursive(neighbour_mask, equal_depth_mask, row, col - 1)

		if col + 1 < self.size and not neighbour_mask[row, col + 1] and equal_depth_mask[row, col + 1]:
			neighbour_mask[row, col + 1] = True
			self.check_neighbour_recursive(neighbour_mask, equal_depth_mask, row, col + 1)

		return

	def find_all_equal_neighbours(self, start_row, start_col):
		'''
			def find_all_equal_neighbours(self, start_row, start_col)

			create mask for all the neighbours with equal depth as starting index
		'''

		depth_start = self.current_depth[start_row, start_col]
		equal_depth_mask = self.current_depth == depth_start
		equal_depth_index = np.where(equal_depth_mask)

		neighbour_mask = np.zeros((self.size, self.size), dtype = bool)
		neighbour_mask[start_row, start_col] = True

		self.check_neighbour_recursive(neighbour_mask, equal_depth_mask, start_row, start_col)

		return neighbour_mask

	def find_leak_count(self, neighbour_mask):
		'''
			def find_leak_count(self, neighbour_mask)

			calculate the total number of outflows from the set of neighbours
		'''
		
		outflow_total = 0
		leak_mask = np.zeros(neighbour_mask.shape, dtype = bool)
		leak_size = np.zeros(neighbour_mask.shape)

		# to avoid an error when there's only one element in list
		row_index, col_index = np.where(neighbour_mask)
		for i, j in zip(row_index, col_index):
			outflow = self.no_out_flow(i, j, count = True)
			if outflow > 0:
				leak_mask[i, j] = True
				leak_size[i, j] = outflow
			outflow_total += outflow

		return outflow_total, leak_mask, leak_size

	def update_equal_leaks(self, rate_total, leak_count, leak_mask, leak_size):
		'''
			def reset_leaks(self, rate_total, leak_count, leak_mask, leak_size)

			update the water rates recursively for leaks due to neighbours
		'''
		next_rate = rate_total / leak_count

		row_index, col_index = np.where(leak_mask)
		for i, j in zip(row_index, col_index):
			self.update_water_rate(next_rate * leak_size[i,j], i, j)

	def update_all_equal_neighbours(self):
		'''
			def update_all_equal_neighbours(self)

			find and update the water_rate for all the neighbours with equal depth
		'''

		updated_mask = np.zeros((self.size, self.size), dtype = bool)
		for i in range(self.size):
			for j in range(self.size):
				if not updated_mask[i, j]:
					neighbour_mask = self.find_all_equal_neighbours(i, j)
					# need leak_count determined by all neighbours find outflow
					leak_count, leak_mask, leak_size = self.find_leak_count(neighbour_mask)
					rate_total = self.water_rate[neighbour_mask].sum()
					if rate_total > 0 and leak_count > 0:
						# handle the leak case
						self.water_rate[neighbour_mask] = 0.
						self.update_equal_leaks(rate_total, leak_count, leak_mask, leak_size)
					elif rate_total > 0:
						self.update_equal_levels(neighbour_mask)

	def update_min_fill(self, min_fill, row, col):
		'''
			def update_min_fill(self, min_fill, row, col)

			update the minimum fill needed until the current cell reaches
			the closest neighouring levels
		'''
		this_depth = self.current_depth[row, col]

		up_min_fill, down_min_fill, left_min_fill, right_min_fill = 0, 0, 0, 0

		def replace_if_zero(x, replace):
			if x <= 0:
				return replace
			else:
				return x

		if row - 1 >= 0:
			up_min_fill = this_depth - self.current_depth[row - 1, col]
		up_min_fill = replace_if_zero(up_min_fill, self.size ** 2)
		if row + 1 < self.size:
			down_min_fill = this_depth - self.current_depth[row + 1, col]
		down_min_fill = replace_if_zero(down_min_fill, self.size ** 2)
		if col - 1 >= 0:
			left_min_fill = this_depth - self.current_depth[row, col - 1]
		left_min_fill = replace_if_zero(left_min_fill, self.size ** 2)
		if col + 1 < self.size:
			right_min_fill = this_depth - self.current_depth[row, col + 1]
		right_min_fill = replace_if_zero(right_min_fill, self.size ** 2)

		min_value = min(up_min_fill, down_min_fill, left_min_fill, right_min_fill)
		min_fill[row, col] = min_value
		return 

	def calc_iter_time(self):
		'''
			def calc_iter_time(self)

			calculate the time that the current state will run until an update 
			is needed
		'''
		rate_mask = self.water_rate > 0
		min_fill = np.zeros((self.size, self.size))
		for i in range(self.size):
			for j in range(self.size):
				if rate_mask[i, j]:
					self.update_min_fill(min_fill, i, j)

		min_time = min( min_fill[rate_mask] / self.water_rate[rate_mask] )
		return min_time

	def update_next_iter(self, print_opt = False):
		'''
			def update_next_iter(self)

			updates one iteration of water flow

			note - need to clear self.water_rate to start
		'''
		self.water_rate = np.zeros((self.size, self.size))

		self.update_flow_dir()
		self.update_water_rate()
		prev_water_rate = np.zeros(self.water_rate.shape)

		while not (self.water_rate == prev_water_rate).all():
			prev_water_rate = self.water_rate.copy()
			self.update_all_equal_neighbours()

		self.iter_time = self.calc_iter_time()
		self.time_passed += self.iter_time
		self.current_depth = self.current_depth - self.water_rate * self.iter_time

		if print_opt:
			test_well.print_status()

	def print_status(self):
		'''
			def print_status(self)

			prints the current status of the well
		'''

		print '\n'
		print 'Time Elapsed: ', self.time_passed
		print 'Time In Last Iteration: ', self.iter_time
		print '\n'
		print 'Previous Water Rising Rate: \n'
		print self.water_rate
		print '\n'
		print 'Current Depth: \n'
		print self.current_depth

	def terminate_fill(self):
		'''
			def terminate_fill(self)

			determine if the process can be terminated as the final cell 
			is filling water
		'''
		if self.current_depth[-1, -1] == self.current_depth[-1, -2] or \
			self.current_depth[-1, -1] == self.current_depth[-2, -1] or \
			self.water_rate[-1, -1] > 0.:
			return True
		else:
			return False



if __name__ == '__main__':
	np.set_printoptions(precision = 4)
	init_depth = np.array([1, 5, 27, 22, 28, 40, 14, 39, 13, 17, 30, 41, 12, 2, 32, 35, 24, 25, 19, 47, 34, 16, 33, 10, 42, 7, 44, 18, 3, 8, 45, 37, 4, 21, 20, 15, 46, 38, 6, 26, 48, 49, 9, 23, 31, 29, 11, 36, 43]).reshape((7, 7))
	# init_depth = np.array([1, 24, 21, 20, 17, 23, 2, 3, 4, 16, 22, 5, 6, 7, 13, 19, 8, 9, 10, 12, 18, 15, 14, 11, 25]).reshape((5, 5))

	test_well = well(size = 7, seed = 1, init_depth = init_depth)
	# test_well = well(size = 7)
	# test_well = well(size = 5, seed = 1, init_depth = init_depth)
	test_well.print_status()

	# this could be one iteration late - but that's fine
	while not test_well.terminate_fill():
		test_well.update_next_iter(print_opt = True)

	print '\n'
	print 'Initial Depth: \n'
	print test_well.init_depth












