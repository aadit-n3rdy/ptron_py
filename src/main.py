import numpy as np
import pygame as pg

lrate = 0.5


def sigmoid(z: float):
    return 1 / (1 + np.e ** -z)


def sigmoid_p(z: float):
    return sigmoid(z) * (1 - sigmoid(z))


class Layer:
    inpCount: int
    weights = np.ndarray(1)
    z: np.array
    a: np.array
    da: np.ndarray

    def __init__(self, prevLayerCount: int, curLayerCount: int):
        self.inpCount = prevLayerCount
        self.z = np.array([1.] * 10)
        self.a = np.array([1.] * 10)
        self.weights = 5*np.random.random_sample((curLayerCount, prevLayerCount)) -5
        self.da = np.zeros(curLayerCount)

    def calc(self, prevLayer: np.array):
        if prevLayer.size != self.inpCount:
            print("Invalid input passed")
            exit(0)
        self.z = self.weights.dot(np.ndarray(prevLayer)).tonp.array()
        for i in range(0, self.z.size):
            self.a[i] = sigmoid(self.z[i])
        return self.a

    def cost_func(self, ideal: np.array):
        if ideal.size != self.a.size:
            exit(1)
        err = 0
        for i in range(0, ideal.size):
            err += abs(ideal[i] ** 2 - self.a[i] ** 2)

    def derivative_final(self, ideal: np.array):
        if ideal.size != self.a.size:
            exit(1)
        for i in range(0, self.a.size):
            self.da[i] = ideal[i] - 1
        return self.da

    def derivative_hidden(self, next_z: np.array, next_da: np.array, next_weights: np.ndarray):
        for i in range(0, self.a.size):
            self.da[i] = 0
            for j in range(0, next_da.size):
                self.da[i] += next_weights[j][i] * sigmoid_p(next_z[j]) * next_da[j]
        return self.da

    def weight_adjust(self, previousLayer: np.array):
        if previousLayer.size != self.inpCount:
            exit(1)
        dw = 0
        for i in range(0, previousLayer.size):
            for j in range(0, self.da.size):
                dw += self.da[j] * sigmoid_p(self.z[j] * previousLayer[i])
                self.weights[j][i] -= dw * lrate


class Perceptron:
    shape: np.ndarray
    learning_rate = 0.5
    layers: np.ndarray

    def __init__(self, shp: np.ndarray):
        self.shape = shp
        self.layers = np.ndarray(self.shape.size - 1, dtype=Layer)
        for i in range(1, self.shape.size):
            self.layers[i - 1] = Layer(self.shape[i - 1], self.shape[i])

    def learn(self, inp: np.ndarray, expected: np.ndarray):
        if expected.size != self.shape[self.shape.size - 1]:
            print("Invalid size")
        else:
            self.calc(inp)
            self.layers[-1].derivative_final(expected)
            for i in range(self.layers.size - 2, -1, -1):
                self.layers[i].derivative_hidden(self.layers[i + 1].z, self.layers[i + 1].da,
                                                 self.layers[i + 1].weights)
            self.layers[0].weight_adjust(inp)
            for i in range(1, self.layers.size):
                self.layers[i].weight_adjust(self.layers[i + 1].a)

    def calc(self, inp: np.ndarray):
        if inp.size != self.shape[0]:
            print("Invalid input size")
        else:
            self.layers[0].calc(inp)
            for i in range(1, self.layers.size):
                self.layers[i].calc(self.layers[i - 1].a)
            return self.layers[self.layers.size - 1].a


randomize = lambda x: x + np.random.randint(-10, 10) * 0.01


def dist(p1: (float, float), p2: (float, float)):
    x1, y1 = p1
    x2, y2 = p2
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


class Rocket:
    ptron: Perceptron
    position: (0., 0.)
    obs: (float, float)
    trg: (float, float)
    aliveTime: 0
    doneTime: 0
    velocity: (0., 0.)
    timeFactor = 0.01
    ptron_scale_factor = 50
    alive = True
    done = False

    def __init__(self, pos, obsP: (float, float), trg: (float, float), parent_ptron: Perceptron = None):
        self.obs = obsP
        self.position = pos
        self.trg = trg
        if parent_ptron is None:
            self.ptron = Perceptron(np.ndarray([6, 4, 2]))
        else:
            self.ptron = parent_ptron
            for i in range(0, self.ptron.layers.size):
                self.ptron.layers[i].weights = randomize(self.ptron.layers[i].weights)

    def display(self, surface):
        if self.alive:
            pg.draw.circle(surface, (255, 255, 255), self.position, 5, 2)
        elif self.done:
            pg.draw.circle(surface, (0, 255, 0), self.position, 5, 2)
        else:
            pg.draw.circle(surface, (255, 0, 0), self.position, 5, 2)

    def update(self):
        if not (self.done or not self.alive):
            pos = self.position
            trg = self.trg
            obsL = self.obs
            force = Perceptron.calc(
                np.ndarray([trg[0] - pos[0], trg[1] - pos[1],
                            obsL[0] - pos[0], obsL[1] - pos[1],
                            self.velocity[0], self.velocity[1]]))
            self.velocity[0] += force[0] * self.timeFactor * self.ptron_scale_factor
            self.velocity[1] += force[1] * self.timeFactor * self.ptron_scale_factor
            self.position[0] += self.velocity[0] * self.timeFactor
            self.position[1] += self.velocity[1] * self.timeFactor
            self.aliveTime += 1
            self.doneTime += 1
        elif self.alive:
            self.aliveTime += 1


pg.init()
size = (width, height) = (800, 600)
obs = (700., 300.)
target = (800., 300.)
spawn = (0., 300.)
obsrad = 50
screen = pg.display.set_mode(size)
rockets = [Rocket(spawn, obs, target), Rocket(spawn, obs, target), Rocket(spawn, obs, target)]
while True:
    screen.fill((0, 0, 0))
    pg.draw.circle(screen, (255, 0, 0), obs, obsrad)
    pg.draw.circle(screen, (0, 255, 0), target, obsrad)
    for event in pg.event.get():
        if event == pg.QUIT:
            exit(0)
    for r in rockets:
        r.update()
    for r in rockets:
        if dist(r.position, obs) < obsrad:
            r.alive = False
        elif dist(r.position, target) < 50:
            # r.done = True
            # RESET
            temp = r
            rockets = [Rocket(spawn, obs, target, temp.ptron),
                       Rocket(spawn, obs, target, temp.ptron),
                       Rocket(spawn, obs, target, temp.ptron)]

        r.display(screen)

    pg.display.flip()
