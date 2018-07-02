import matplotlib.pyplot as plt
from array import array

for size in [20, 40, 60, 80]:
    input_file = open('data/results_{}.out'.format(size), 'rb')
    float_array = array('d')
    float_array.fromstring(input_file.read())
    sz = len(float_array)
    plt.plot(range(1, sz + 1), float_array, label='{}'.format(size))

plt.grid()
plt.legend(title='Число программистов в выборке')
plt.ylabel('Доля содержащих правильный ответ')
plt.xlabel('Число рассматриваемых ответов')
plt.savefig('/home/egor/Work/prezo/spring-2018/graphs.png')
plt.show()
