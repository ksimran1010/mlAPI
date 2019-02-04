from numpy import *


class LetsGo:

    def __init__(self, iterations):
        self.iter = iterations

    def error(self, b, m, points):
        totalError = 0
        for i in range(0, len(points)):
            x = points['X'][i]
            y = points['Y'][i]
            totalError += (y - (m * x + b)) ** 2
        return totalError / float(len(points))

    def step_gradient(self, b_current, m_current, points, learningRate):
        b_gradient = 0
        m_gradient = 0
        # print(len(points ),'///////////////////////////////////////////////')
        N = float(len(points))
        for i in range(0, len(points)):
            x = points[i, 0]
            y = points[i, 1]
            b_gradient += -(2 / N) * (y - ((m_current * x) + b_current))
            m_gradient += -(2 / N) * x * (y - ((m_current * x) + b_current))
        new_b = b_current - (learningRate * b_gradient)
        new_m = m_current - (learningRate * m_gradient)
        return [new_b, new_m]

    def gradient_descent_iterator(self, points, starting_b, starting_m, learning_rate, num_iterations):
        b = starting_b
        m = starting_m
        for i in range(num_iterations):
            b, m = self.step_gradient(b, m, array(points), learning_rate)
        return [b, m]

    def run(self):
        import pandas as pd
        dataset = pd.read_csv('data/regdata.csv')

        # initializing our inputs and outputs
        points = dataset[['X', 'Y']]

        print('this is the given data ', points)
        learning_rate = 0.0001
        initial_b = 0  # initial y-intercept guess
        initial_m = 0  # initial slope guess
        num_iterations = self.iter
        print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m,
                                                                                  self.error(initial_b, initial_m,
                                                                                             points)))

        [b, m] = self.gradient_descent_iterator(points, initial_b, initial_m, learning_rate, num_iterations)
        print(
            "After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, self.error(b, m, points)))

# if __name__=='__main__':
#     runObj = LetsGo(1000)
#     runObj.run()
