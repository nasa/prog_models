# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

class ProgressBar():
    def __init__(self, n, prefix='', suffix='', decimals=1, print_length=100, fill='█', print_end = " "):
        self.n = n
        self.prefix = prefix
        self.suffix = suffix
        self.decimals = decimals
        self.print_length = print_length
        self.fill = fill
        self.print_end = print_end
        print('\r%s |%s| %s%% %s' % (self.prefix, self.fill * 0 + '-' * (self.print_length - 0), 0, self.suffix), end = self.print_end)

    def __call__(self, iteration):
        percent = ("{0:." + str(self.decimals) + "f}").format(100 * (iteration / float(self.n)))
        filledLength = int(self.print_length * iteration // self.n)
        bar = self.fill * filledLength + '-' * (self.print_length - filledLength)
        print('\r%s |%s| %s%% %s' % (self.prefix, bar, percent, self.suffix), end = self.print_end)
        # Print New Line on Complete
        if iteration == self.n:
            print('')

# if __name__ == '__main__':
#     from time import sleep
#     pb = ProgressBar(10, 'Progress',)
#     for i in range(10):
#         sleep(.25)
#         pb(i)
