from .algorithms import Perceptron, PA, AveragedPerceptron, Multi_Perceptron, Multi_PA
import Utils

# Orchastra inputs to methods
def run():
    functions = {
        "1a": p1a,
        "1b": p1b,
        "1c": p1c,
        "1d": p1d,
        "1d2": p1d2,
        "2a": p2a,
        "2a2": p2a2,
        "2b": p2b,
        "2b2": p2b2,
        "2c": p2c,
        "2c2": p2c2,
        "2d": p2d,
        "2d2": p2d2,
    }
    print("Choose problem (example: 1a):")
    command = input()
    if command in functions:
        print("Executing " + command + " ...")
        functions[command]()
        print("Finished")
    else:
        print("Invalid command, now exiting...")

def p1a():
    ret = Perceptron.run(iteration=50)
    mistake_list = ret[1]
    Utils.generate_graph(
        x = range(1, 51),
        y = mistake_list,
        title = "Binary Perceptron Mistakes",
        xlabel = "Iteration",
        ylabel = "Mistakes"
    )

    ret = PA.run(iteration = 50)
    mistake_list = ret[1]
    Utils.generate_graph(
        x = range(1, 51),
        y = mistake_list,
        title = "Binary PA Mistakes",
        xlabel = "Iteration",
        ylabel = "Mistakes"
    )
    return

def p1b():
    ret = Perceptron.run(iteration=20)
    train_accuracy_list = ret[2]
    test_accuracy_list = ret[3]
    Utils.generate_graph(
        x = range(1, 21),
        y = train_accuracy_list,
        title = "Binary Perceptron Training Accuracy",
        xlabel = "Iteration",
        ylabel = "Accuracy"
    )
    Utils.generate_graph(
        x = range(1, 21),
        y = test_accuracy_list,
        title = "Binary Perceptron Test Accuracy",
        xlabel = "Iteration",
        ylabel = "Accuracy"
    )

    #############
    ret = PA.run(iteration=20)
    train_accuracy_list = ret[2]
    test_accuracy_list = ret[3]
    Utils.generate_graph(
        x = range(1, 21),
        y = train_accuracy_list,
        title = "Binary PA Training Accuracy",
        xlabel = "Iteration",
        ylabel = "Accuracy"
    )
    Utils.generate_graph(
        x = range(1, 21),
        y = test_accuracy_list,
        title = "Binary PA Test Accuracy",
        xlabel = "Iteration",
        ylabel = "Accuracy"
    )
    return 

def p1c():
    ret = Perceptron.run(iteration=20)
    test_accuracy_list = ret[3]
    Utils.generate_graph(
        x = range(1, 21),
        y = test_accuracy_list,
        title = "Binary Perceptron Training Accuracy",
        xlabel = "Iteration",
        ylabel = "Accuracy"
    )

    ret = AveragedPerceptron.run(iteration=20)
    test_accuracy_list = ret[1]
    Utils.generate_graph(
        x = range(1, 21),
        y = test_accuracy_list,
        title = "Binary Averaged Perceptron Accuracy",
        xlabel = "Iteration",
        ylabel = "Accuracy"
    )
    return 

def p1d():
    ret = Perceptron.incrementRun(iteration=20, size=100)
    data_size = ret[1]
    test_accuracy_list = ret[2]
    Utils.generate_graph(
        x = data_size,
        y = test_accuracy_list,
        title = "Binary Perceptron Training Accuracy",
        xlabel = "Iteration",
        ylabel = "Accuracy"
    )
    return 

def p1d2():
    ret = PA.incrementRun(iteration=20, size=100)
    data_size = ret[1]
    test_accuracy_list = ret[2]
    Utils.generate_graph(
        x = data_size,
        y = test_accuracy_list,
        title = "Binary PA Training Accuracy",
        xlabel = "Iteration",
        ylabel = "Accuracy"
    )
    return 

def p2a():
    ret = Multi_Perceptron.run(iteration=50)
    mistake_list = ret[1]
    Utils.generate_graph(
        x = range(1, 51),
        y = mistake_list,
        title = "Multi-class Perceptron Mistakes",
        xlabel = "Iteration",
        ylabel = "Mistakes"
    )
    return

def p2a2():
    ret = Multi_PA.run(iteration=50)
    mistake_list = ret[1]
    Utils.generate_graph(
        x = range(1, 51),
        y = mistake_list,
        title = "Multi-class PA Mistakes",
        xlabel = "Iteration",
        ylabel = "Mistakes"
    )

def p2b():
    ret = Multi_Perceptron.run(iteration=20)
    train_accuracy_list = ret[2]
    test_accuracy_list = ret[3]
    Utils.generate_graph(
        x = range(1, 21),
        y = train_accuracy_list,
        title = "Multi-class Perceptron Training Accuracy",
        xlabel = "Iteration",
        ylabel = "Accuracy"
    )
    Utils.generate_graph(
        x = range(1, 21),
        y = test_accuracy_list,
        title = "Multi-class Perceptron Test Accuracy",
        xlabel = "Iteration",
        ylabel = "Accuracy"
    )
    return

def p2b2():
    #############
    ret = Multi_Perceptron.run(iteration=20)
    train_accuracy_list = ret[2]
    test_accuracy_list = ret[3]
    Utils.generate_graph(
        x = range(1, 21),
        y = train_accuracy_list,
        title = "Multi-class PA Training Accuracy",
        xlabel = "Iteration",
        ylabel = "Accuracy"
    )
    Utils.generate_graph(
        x = range(1, 21),
        y = test_accuracy_list,
        title = "Multi-class PA Test Accuracy",
        xlabel = "Iteration",
        ylabel = "Accuracy"
    )
    return 

def p2c():
    print("Not Implemented")
    return

def p2c2():
    print("Not Implemented")
    return

def p2d():
    ret = Multi_Perceptron.incrementRun(iteration=20, size=100)
    data_size = ret[1]
    test_accuracy_list = ret[2]
    Utils.generate_graph(
        x = data_size,
        y = test_accuracy_list,
        title = "Multi-class Perceptron Training Accuracy",
        xlabel = "Iteration",
        ylabel = "Accuracy"
    )
    return 

def p2d2():
    ret = Multi_PA.incrementRun(iteration=20, size=100)
    data_size = ret[1]
    test_accuracy_list = ret[2]
    Utils.generate_graph(
        x = data_size,
        y = test_accuracy_list,
        title = "Multi-class PA Training Accuracy",
        xlabel = "Iteration",
        ylabel = "Accuracy"
    )
    return 