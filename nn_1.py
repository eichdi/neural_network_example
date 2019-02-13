import numpy as np
import random as rnd
import csv

class Neuron:
  def __init__(self, weights, output = 0, diff = 0): # diff - ошибка нейрона
    self.output = output # выходное значение нейрона
    self.diff = diff

    if np.count_nonzero(weights) == 0:
      self.weights = np.array([rnd.random() for weight in weights])
    else:
      self.weights = weights

  def __activtion_function(self, x): # функция активации
    return 1 / (1 + np.exp(-1 * x))
    # return (np.exp(2 * x) - 1)/(np.exp(2 * x) + 1)
    # return math.tanh(x)

  def __activation_function_derivative(self, x): # производная функции активации, использующаяся при обучении
    return self.__activtion_function(x) * (1 - self.__activtion_function(x))

  def activate(self, inputs): # функция, вычисляющая выходное значение нейрона
    inputs_sum = np.dot(self.weights, inputs) # перемножение 'матриц' весов и входных значений
    self.output = self.__activtion_function(inputs_sum)

  def activate_derivative(self, inputs): # функция, вычисляющая произодную от взвешенной суммы, используется при обучении
    inputs_sum = np.dot(self.weights, inputs)
    return self.__activation_function_derivative(inputs_sum)

class Layer:
  def __init__(self, name, neurons):
    self.name = name
    self.neurons = neurons

class Network:
  def __init__(self, layers):
    self.layers = layers

  def activate(self): # функция, запускающая процесс вычисления значений нейронов в сети для каждого слоя
    for i, layer in enumerate(self.layers): # layer содержит текущий слой
      if not i: # skip first layer, т.к. первый слой по сути содержит только входные значения
        continue
      previous_layer = self.layers[i - 1] # переменная предыдущего слоя для получения выходных значений нейронов этого слоя

      for neuron in layer.neurons: # перебираем каждый нейрона в текущем слое и активируем его входными значениями предыдущего слоя
        inputs = np.array([neuron.output for neuron in previous_layer.neurons])
        neuron.activate(inputs)

  def get_result(self): # функция получения результата работы нейронной сети
    result_layer = self.layers[-1]
    result = []
    for neuron in result_layer.neurons:
      result.append(neuron.output)
    return result
    # return result[0] if len(result) == 1 else result

  def set_inputs(self, inputs): # функция для установки входных значений нейронной сети
    input_layer = self.layers[0]
    for i, neuron in enumerate(input_layer.neurons):
      input_layer.neurons[i].output = inputs[i]

  # back propagation
  def study(self, trainingInputs): # функция обучения нейронной сети
    iterationsCount = 0 # количество итераций, которое пришлось сделать во время обучения

    # цикл do while
    while True:
      errorsSum = 0 # содержит сумму модулей разниц, используется для определения момента завершения обучения

      for inputs in trainingInputs: # выборка содержит обучающую выборку с эталонными результатами
        self.set_inputs(inputs) # устанавливаем входные значения нейронной сети на первый слой
        self.activate()

        # расчет разниц
        result_layer = self.layers[-1] # переменная содержит последний слой, от которого начнётся рассчёт ошибок
        for i, neuron in enumerate(result_layer.neurons):
          neuron.diff = inputs[-1] - self.get_result()[i] # вычисляем разницу d = e - y
          errorsSum += abs(neuron.diff) # суммирование модулей разниц

        for i, layer in reversed(list(enumerate(self.layers))): # обходим слои нейронной сети с конца, последнего слоя для расчёта ошибок для каждого нейрона
          if i == len(self.layers) - 1: # skip result layer, т.к. разница для него уже посчитана
            continue
          previous_layer = self.layers[i + 1] # в переменной храниться слой, идущий после текущего по направлению к последнему (для предпоследнего слоя это будет последний и т.д.)

          for neuron in layer.neurons: # обнуляем значения ошибок в нейронах текущего слоя для последующего перерасчёта
            neuron.diff = 0

          for neuron in previous_layer.neurons: # перебираем нейроны предшествующего текущему слоя
            for k, weight in enumerate(neuron.weights): # перебираем веса нейронов предшествующего слоя
              layer.neurons[k].diff += weight * neuron.diff # делаем перерасчёт ошибкок нейронов текущего слоя = сумма произведений весов и оишбок нейронов предшествующего слоя

        # перерасчет весов
        for i, layer in enumerate(self.layers):
          if not i: # skip first layer
            continue
          previous_layer = self.layers[i - 1]

          for neuron in layer.neurons: # перебираем нейроны текущего слоя
            inputs = np.array([neuron.output for neuron in previous_layer.neurons]) # содержит входные значения для текущего слоя (выходные значения нейронов для предыдущего слоя)
            derivative = neuron.activate_derivative(inputs) # производная от взвешенной суммы

            for k, weight in enumerate(neuron.weights): # перебираем веса нейрона текущего слоя
              k_input = previous_layer.neurons[k].output # k-ое входное значение для нейрона
              neuron.weights[k] = neuron.weights[k] + neuron.diff * derivative * k_input * 0.8 # weight'k = weightk + neuron diff + F'(Sk) * inputk * a (скорость обучения)

      iterationsCount += 1

      # результат работы сети из-за особенностей функции активации может находиться в пределах (0, 1)
      # можно ввести условность, что зеленый цвет свечи соответсвует 0,95 красный - 0,05 отсутствие тела - 0,45
      # когда сумма модулей разниц работ сети и эталонных значений принимает маленькое значение, то сеть можно считать более-менее обученной
      print(errorsSum)
      if abs(errorsSum) < 0.05:
        print("Iterations:", iterationsCount)
        break

def read_csv(csv_path):
  result = []

  with open(csv_path, "r") as file_obj:
    reader = csv.reader(file_obj)
    # next(reader, None)  # skip the headers
    for row in reader:
      QUONTION = str(row[0]).split(";")
      OPEN = float(QUONTION[2])
      CLOSE = float(QUONTION[5])

      result.append([OPEN, CLOSE])

  return result

def main():
  network = Network([
    Layer("Input layer", [
      Neuron(np.ones(1)), # начальный вес нейрона входного слоя равен 1
      Neuron(np.ones(1))
    ]),
    Layer("First hidden layer", [
      Neuron(np.zeros(2)), # начальным весам нейронов внутренних и выходного слоёв присваиваются рандомные значения в контрукторе
      Neuron(np.zeros(2)),
      Neuron(np.zeros(2)),
      Neuron(np.zeros(2)),
      Neuron(np.zeros(2)),
      Neuron(np.zeros(2)),
      Neuron(np.zeros(2)),
      Neuron(np.zeros(2)),
      Neuron(np.zeros(2)),
      Neuron(np.zeros(2)),
      Neuron(np.zeros(2)),
      Neuron(np.zeros(2)),
      Neuron(np.zeros(2)),
      Neuron(np.zeros(2)),
      Neuron(np.zeros(2)),
      Neuron(np.zeros(2)),
      Neuron(np.zeros(2)),
      Neuron(np.zeros(2)),
      Neuron(np.zeros(2)),
      Neuron(np.zeros(2)),
      Neuron(np.zeros(2)),
      Neuron(np.zeros(2)),
      Neuron(np.zeros(2)),
      Neuron(np.zeros(2)),
      Neuron(np.zeros(2)),
      Neuron(np.zeros(2)),
      Neuron(np.zeros(2)),
      Neuron(np.zeros(2)),
      Neuron(np.zeros(2)),
      Neuron(np.zeros(2))
    ]),
    Layer("Second hidden layer", [
      Neuron(np.zeros(30)),
      Neuron(np.zeros(30)),
      Neuron(np.zeros(30)),
      Neuron(np.zeros(30)),
      Neuron(np.zeros(30)),
      Neuron(np.zeros(30)),
      Neuron(np.zeros(30)),
      Neuron(np.zeros(30)),
      Neuron(np.zeros(30)),
      Neuron(np.zeros(30)),
      Neuron(np.zeros(30)),
      Neuron(np.zeros(30)),
      Neuron(np.zeros(30)),
      Neuron(np.zeros(30)),
      Neuron(np.zeros(30)),
      Neuron(np.zeros(30)),
      Neuron(np.zeros(30)),
      Neuron(np.zeros(30)),
      Neuron(np.zeros(30)),
      Neuron(np.zeros(30))
    ]),
    Layer("Output layer", [
      Neuron(np.zeros(20))
    ])
  ])

  trainingInputs = read_csv("EURRUB.csv")
  network.study(trainingInputs)

main()