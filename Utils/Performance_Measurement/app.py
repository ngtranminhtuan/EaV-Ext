from tabulate import tabulate
import time

class Logging:
    def __init__(self, debug=True):
        self.debug = debug
        self.data = {}
        self.sum_data = {}
        self.counter_data = {}
        self.counter = 0
    
    def start(self, name):
        if self.debug:
            if self.counter > 3:
                t = time.time()
                self.data[name] = [t]

                if name not in self.sum_data:
                    self.sum_data[name] = [0]
                    self.counter_data[name] = [0]

    def end(self, name):
        if self.debug:
            if self.counter > 3:
                elapsed_time = round(time.time() - self.data[name][0], 3)
                self.data[name][0] = elapsed_time

                # Add into sum_data
                self.sum_data[name][0] += elapsed_time
                self.counter_data[name][0] += 1

    def display(self, name, content):
        if self.debug:
            if name not in self.sum_data:
                self.sum_data[name] = [0]
            self.sum_data[name][0] = content
            self.data[name] = [content]

    def add(self, name, time):
        if self.debug:
            if self.counter > 3:
                self.data[name][0] += round(time, 3)

    def sub(self, name, time):
        if self.debug:
            if self.counter > 3:
                self.data[name][0] -= round(time, 3)

    def mul(self, name, ratio):
        if self.debug:
            if self.counter > 3:
                self.sum_data[name][0] += (self.data[name][0]*ratio) - (self.data[name][0])
                self.data[name][0] *= ratio
                self.data[name][0] = round(self.data[name][0], 3)

    def div(self, name, ratio):
        if self.debug:
            if self.counter > 3:
                self.data[name][0] /= ratio
                self.data[name][0] = round(self.data[name][0], 3)

    def print_result(self):
        if self.debug:
            print(tabulate(self.data,headers=self.data.keys(), tablefmt="fancy_grid"))
            self.counter += 1

    def print_mean_result(self):
        if self.debug:
            mean_data = {}
            for key in self.sum_data.keys():
                mean_data[key] = [0]
                if isinstance(self.sum_data[key][0], int) or isinstance(self.sum_data[key][0], float):
                    mean_data[key][0] = round(self.sum_data[key][0] / self.counter_data[key][0], 3)
                elif isinstance(self.sum_data[key][0], str):
                    mean_data[key][0] =self.sum_data[key][0]
            print(tabulate(mean_data,headers=mean_data.keys(), tablefmt="fancy_grid"))
            self.counter += 1

if __name__ == "__main__":
    log = Logging(True)

    for i in range(10):
        log.display("Counter", str(i))
        log.start("Inference time")
        print("Test function 1 running...")
        time.sleep(1)
        log.end("Inference time")

        log.start("Postprocess time")
        print("Test function 2 running...")
        time.sleep(2)
        log.end("Postprocess time")

        log.add("Postprocess time", 1)
        log.sub("Postprocess time", 2)
        log.mul("Postprocess time", 3)
        log.div("Postprocess time", 2)

        log.print_result()


    # print("Average Time")
    # log.print_mean_result()

    
    



