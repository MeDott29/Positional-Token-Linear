# linear equation
import math
import numpy as np

x=np.arange(0, 100)
n=len(x)
m=10
b=0

y = m*x + b
print(y)


matrix_y = np.array([y]).T
print(matrix_y/5)

# gradual normalization plot

# generate a four letter word

fourletterword="bruh"

#generate another four letter word
anotherfourletterword="breh"
#generate another four letter word
anotheranotherfourletterword="brebreh"

# print the word
print(f"{fourletterword}-----------------------------------------------")
# distribute the matrix into five columns
for i in range(len(matrix_y)):
    print(f"{matrix_y[i]}", end="")
    if (i+1) % 10 == 0:
        print("") # print out an ASCII informed terminal plot 

# ASCII art terminal plot
print("\n")
print("Terminal Plot:")
print("-" * 80)

max_val = np.max(matrix_y)
height = 20  # Height of the plot
width = 80   # Width of the plot

# Create the plot array
plot = [[' ' for _ in range(width)] for _ in range(height)]

# Scale the values to fit the height
scaled_y = (matrix_y / max_val * (height-1)).astype(int)

# Fill the plot array
for x in range(min(width, len(scaled_y))):
    y = scaled_y[x][0]
    plot[height-1-y][x] = '█'

# Add axis
for i in range(height):
    plot[i][0] = '│'
plot[height-1] = ['─' for _ in range(width)]
plot[height-1][0] = '└'

# Print the plot with some pizzazz
print("\033[32m")  # Green text
for row in plot:
    print(''.join(row))
print("\033[0m")   # Reset color

# Add some cyberpunk flair
print("\n")
print("\033[35m╔" + "═" * 78 + "╗\033[0m")
print("\033[35m║\033[0m" + f"{'NEURAL INTERFACE ACTIVATED':^78}" + "\033[35m║\033[0m")
print("\033[35m╚" + "═" * 78 + "╝\033[0m")
