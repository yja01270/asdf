
02 - Simple Liner Regression

In [1]:
import tensorflow as tf
import numpy as np
Hypothesis and Cost
Hypothesis
$$ H(x) = Wx + b $$
Cost
$$ cost(W)=\frac { 1 }{ m } \sum _{i=1}^{m}{ { (W{ x }^{ i }-y^{ i } })^{ 2 } }  $$
Which Hypothesis is better ?
cost function: 편차제곱의 평균
learning의 목표는 cost(W,b)를 최소화하는 (W,b)를 구하는 것
In [1]:
import tensorflow as tf

x_data = [1, 2, 3]
y_data = [1, 2, 3]


W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

hypothesis = W * x_data + b

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.initializers.global_variables())

for step in range(1000):
    sess.run(train)
    if step % 30 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))
0 0.021165216 [0.9768667] [0.1905179]
30 0.00094205124 [0.9643522] [0.08103587]
60 0.00021878286 [0.9828208] [0.03905234]
90 5.0810035e-05 [0.99172115] [0.01881986]
120 1.1800398e-05 [0.9960103] [0.00906954]
150 2.7407639e-06 [0.9980773] [0.00437076]
180 6.3646576e-07 [0.99907345] [0.00210634]
210 1.477923e-07 [0.9995535] [0.0010151]
240 3.4324824e-08 [0.9997848] [0.00048916]
270 7.965265e-09 [0.9998963] [0.00023569]
300 1.8508265e-09 [0.99995] [0.00011358]
330 4.2869885e-10 [0.9999759] [5.47589e-05]
360 9.924861e-11 [0.9999884] [2.6355296e-05]
390 2.3211063e-11 [0.9999944] [1.27257e-05]
420 5.366966e-12 [0.9999973] [6.121506e-06]
450 1.2505552e-12 [0.9999987] [2.9425917e-06]
480 3.2684966e-13 [0.99999934] [1.4405549e-06]
510 6.158037e-14 [0.99999964] [7.570888e-07]
540 6.158037e-14 [0.99999976] [5.504595e-07]
570 4.7369517e-15 [0.99999994] [1.9283151e-07]
600 0.0 [1.] [5.772759e-08]
630 0.0 [1.] [5.772759e-08]
660 0.0 [1.] [5.772759e-08]
690 0.0 [1.] [5.772759e-08]
720 0.0 [1.] [5.772759e-08]
750 0.0 [1.] [5.772759e-08]
780 0.0 [1.] [5.772759e-08]
810 0.0 [1.] [5.772759e-08]
840 0.0 [1.] [5.772759e-08]
870 0.0 [1.] [5.772759e-08]
900 0.0 [1.] [5.772759e-08]
930 0.0 [1.] [5.772759e-08]
960 0.0 [1.] [5.772759e-08]
990 0.0 [1.] [5.772759e-08]
placeholder
In [2]:
import tensorflow as tf

x_data = [1, 2, 3]
y_data = [1, 2, 3]

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = W * X + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))

a = tf.Variable(0.1) 
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.initializers.global_variables())

for step in range(1000):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 30 == 0:
        print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W), sess.run(b))
0 0.23415472 [0.74881464] [0.94065374]
30 0.025842478 [0.81329197] [0.4244312]
60 0.006001683 [0.91002274] [0.20453936]
90 0.0013938443 [0.9566387] [0.09857045]
120 0.00032370738 [0.97910357] [0.04750248]
150 7.5177835e-05 [0.9899297] [0.02289217]
180 1.7459637e-05 [0.995147] [0.01103208]
210 4.0546915e-06 [0.9976613] [0.00531649]
240 9.417765e-07 [0.99887294] [0.00256211]
270 2.1865871e-07 [0.9994569] [0.00123471]
300 5.0772787e-08 [0.9997383] [0.00059503]
330 1.1799879e-08 [0.9998739] [0.0002868]
360 2.7365417e-09 [0.9999392] [0.00013821]
390 6.3606365e-10 [0.99997073] [6.660059e-05]
420 1.4789237e-10 [0.9999859] [3.210142e-05]
450 3.4811858e-11 [0.9999932] [1.548364e-05]
480 8.000711e-12 [0.9999967] [7.480722e-06]
510 1.9184654e-12 [0.9999984] [3.5944997e-06]
540 4.92643e-13 [0.9999992] [1.734835e-06]
570 9.473903e-14 [0.9999996] [8.526869e-07]
600 6.158037e-14 [0.9999997] [5.98374e-07]
630 3.7895614e-14 [0.99999994] [2.7253517e-07]
660 0.0 [1.] [5.7958374e-08]
690 0.0 [1.] [5.7958374e-08]
720 0.0 [1.] [5.7958374e-08]
750 0.0 [1.] [5.7958374e-08]
780 0.0 [1.] [5.7958374e-08]
810 0.0 [1.] [5.7958374e-08]
840 0.0 [1.] [5.7958374e-08]
870 0.0 [1.] [5.7958374e-08]
900 0.0 [1.] [5.7958374e-08]
930 0.0 [1.] [5.7958374e-08]
960 0.0 [1.] [5.7958374e-08]
990 0.0 [1.] [5.7958374e-08]
In [3]:
print(sess.run(hypothesis, feed_dict={X: 5}))
print(sess.run(hypothesis, feed_dict={X: 2.5}))
[5.]
[2.5]

04 - Liner Regression and Minimizing Cost


In [1]:
import tensorflow as tf
import numpy as np
Hypothesis
$$ H(x) = Wx + b $$
cost function
$$ cost(W)=\frac { 1 }{ m } \sum _{ i=1}^{m}{ { (W{ x }^{ i }-y^{ i } })^{ 2 } }  $$
In [2]:
 cost 계산

W = 0
print (((W*1 - 1)**2 + (W*2 - 2)**2 + (W*3 - 3)** 2) / 3)

W = 1
print (((W*1 - 1)**2 + (W*2 - 2)**2 + (W*3 - 3)** 2) / 3)

W = 2
print (((W*1 - 1)**2 + (W*2 - 2)**2 + (W*3 - 3)** 2) / 3)

W = 3
print (((W*1 - 1)**2 + (W*2 - 2)**2 + (W*3 - 3)** 2) / 3)
4.666666666666667
0.0
4.666666666666667
18.666666666666668
In [3]:
data = [
    (1, 1),
    (2, 2),
    (3, 3),
]

def cost_func(w, data):
    s = 0
    m = len(data)
    for v in data:
        s += (w*v[0] - v[1]) ** 2
    return s/m
In [4]:
print( cost_func(0, data) )
print( cost_func(1, data) )
print( cost_func(2, data) )
print( cost_func(3, data) )
4.666666666666667
0.0
4.666666666666667
18.666666666666668
In [5]:
%matplotlib inline
import matplotlib.pyplot as plt

w_vals = range(-3, 4)
cost_vals = [cost_func(w, data) for w in w_vals]

plt.plot(w_vals, cost_vals)
plt.ylabel('cost')
plt.ylabel('W')
plt.grid()

Gradient descent algorithm
cost를 최소화 하는 대표적인 알고리즘
다수의 변수에도 적용이 가능
$$ W\leftarrow W-\alpha \frac { 1 }{ m } \sum _{ i=1 }^{ m }{ (Wx^{ i }-y^{ i })x^{ i } }  $$
liner regression 의 핵심 알고리즘
cost function을 설계할때 반드시 convex function 이어야 한다
In [6]:
import tensorflow as tf

x_data = [1, 2, 3]
y_data = [1, 2, 3]

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = W * X + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

a = tf.Variable(0.1) 
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.initializers.global_variables())

for step in range(1000):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 100 == 0:
        print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W), sess.run(b))
0 0.22860377 [0.75399375] [0.9258992]
100 0.0008292195 [0.96655506] [0.07602829]
200 6.3845923e-06 [0.99706537] [0.00667125]
300 4.917801e-08 [0.99974245] [0.00058543]
400 3.781461e-10 [0.9999774] [5.1359697e-05]
500 3.0932294e-12 [0.999998] [4.5342836e-06]
600 6.158037e-14 [0.99999976] [6.0832514e-07]
700 0.0 [1.] [5.2014947e-08]
800 0.0 [1.] [5.2014947e-08]
900 0.0 [1.] [5.2014947e-08]
In [7]:
 predict

print(sess.run(hypothesis, feed_dict={X: 5}))
print(sess.run(hypothesis, feed_dict={X: 2.5}))
[5.]
[2.5]

03 - Multi-variable Linear Regression


In [1]:
import tensorflow as tf
import numpy as np

tf.__version__
Out[1]:
'1.12.0'
Hypothesis and Cost
$$ H(x) = Wx + b $$
$$ cost(W, b)=\frac { 1 }{ m } \sum _{i=1}^{m}{ { (H{ x }^{ i }-y^{ i } })^{ 2 } }  $$
Simplifed hypothesis
$$ H(x) = Wx $$
$$ cost(W)=\frac { 1 }{ m } \sum _{i=1}^{m}{ { (W{ x }^{ i }-y^{ i } })^{ 2 } }  $$
b를 W 행렬에 넣어 표현할 수 있기 때문에 생략 가능

Cost function
$$ cost(W)=\frac { 1 }{ m } \sum _{i=1}^{m}{ { (W{ x }^{ i }-y^{ i } })^{ 2 } }  $$
W = -1, cost(W) = 18.67 $$ cost(W)=\frac { 1 }{ 3 } ( (-1 * 1 - 1)^2 + (-1 * 2 - 2)^2 + (-1 * 3 - 3)^2) $$

W = 0, cost(W) = 4.67 $$ cost(W)=\frac { 1 }{ 3 } ( (0 * 1 - 1)^2 + (0 * 2 - 2)^2 + (0 * 3 - 3)^2) $$

W = 1, cost(W) = 0 $$ cost(W)=\frac { 1 }{ 3 } ( (1 * 1 - 1)^2 + (1 * 2 - 2)^2 + (1 * 3 - 3)^2) $$

W = 2, cost(W) = 4.67 $$ cost(W)=\frac { 1 }{ 3 } ( (2 * 1 - 1)^2 + (2 * 2 - 2)^2 + (2 * 3 - 3)^2) $$

Cost function in pure Python
In [2]:
import numpy as np

X = [1, 2, 3]
Y = [1, 2, 3]

def cost_func(W, X, Y):
    c = 0
    for i in range(len(X)):
        c += (W * X[i] - Y[i]) ** 2
    return c / len(X)

for feed_W in np.linspace(-3, 5, num=15):
    curr_cost = cost_func(feed_W, X, Y)
    print("%6.3f | %10.5f" % (feed_W, curr_cost))
-3.000 |   74.66667
-2.429 |   54.85714
-1.857 |   38.09524
-1.286 |   24.38095
-0.714 |   13.71429
-0.143 |    6.09524
 0.429 |    1.52381
 1.000 |    0.00000
 1.571 |    1.52381
 2.143 |    6.09524
 2.714 |   13.71429
 3.286 |   24.38095
 3.857 |   38.09524
 4.429 |   54.85714
 5.000 |   74.66667
Cost function in TensorFlow
In [3]:
import tensorflow as tf

X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.Variable(0)

hypothesis = X * W
cost = tf.reduce_mean(tf.square(hypothesis - Y))

sess = tf.Session()
sess.run(tf.initializers.global_variables())

for feed_W in np.linspace(-3, 5, num=15):
    curr_cost = sess.run([cost, W], feed_dict={W: feed_W})
    print("%6.3f | %10.5f" % (feed_W, curr_cost[0]))
-3.000 |   74.00000
-2.429 |   42.00000
-1.857 |   18.00000
-1.286 |   18.00000
-0.714 |    4.00000
-0.143 |    4.00000
 0.429 |    4.00000
 1.000 |    0.00000
 1.571 |    0.00000
 2.143 |    4.00000
 2.714 |    4.00000
 3.286 |   18.00000
 3.857 |   18.00000
 4.429 |   42.00000
 5.000 |   74.00000
In [4]:
import tensorflow as tf
import matplotlib.pyplot as plt

X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.placeholder(tf.float32)

hypothesis = X * W
cost = tf.reduce_mean(tf.square(hypothesis - Y))

sess = tf.Session()
sess.run(tf.initializers.global_variables())

W_val = np.linspace(-3, 5, num=30)
cost_val = []
for feed_W in W_val:
    curr_cost  = sess.run([cost], feed_dict={W: feed_W})
    cost_val.append(curr_cost)

plt.plot(W_val, cost_val, "ro")
plt.ylabel('Cost')
plt.xlabel('W')
plt.show()
<Figure size 640x480 with 1 Axes>
How to minimize cost?
현재 데이터 X와 Y에 대해 W가 1일 때 cost 가 가장 작다
cost 가 최소가 되는 W를 어떻게 찾을 수 있을까?
Gradient descent algorithm
Minimize cost function
used many minimization problems
For a given cost (W, b), it will find W, b to minimize cost
It can be applied to more general function: cost (w1, w2, ...)
How does it work?
Start with initial guesses
Start at 0,0 (or any other value)
Keeping changing $W$ and $b$ a little bit to try and reduce $cost(W,b)$
Each time you change the parameters, you select the gradient which reduces $cost(W,b)$ the most possible
Repeat
Do so until you converge to a local minimum
Has an interesting property
Where you start can determine which minimum you end up


Formal definition
$$ cost(W)=\frac { 1 }{ m } \sum _{i=1}^{m}{ { (W{ x }^{ i }-y^{ i } })^{ 2 } }  $$$$ \Downarrow $$$$ cost(W)=\frac { 1 }{ 2m } \sum _{i=1}^{m}{ { (W{ x }^{ i }-y^{ i } })^{ 2 } }  $$
m 혹은 2m 나누는 것이 cost 최소화에 영향 없음
제곱을 미분할 때, 2가 앞으로 나오면서 공식이 단순하게 되는 효과
Formal definition
$$ cost(W)=\frac { 1 }{ 2m } \sum _{i=1}^{m}{ { (W{ x }^{ i }-y^{ i } })^{ 2 } }  $$$$ W:=W - \alpha\frac{ \partial } {\partial W } cost(W) $$
W = W - 변화량
변화량 = 현 위치(W)에서 비용곡선의 기울기(=미분값) X $\alpha$ 
$\alpha$ : learning rate (시도 간격)
Formal definition
$$ W:=W - \alpha\frac{ \partial } {\partial W } \frac { 1 }{ 2m } \sum _{i=1}^{m}{ { (W{ x }^{ i }-y^{ i } })^{ 2 } } $$$$ W:=W-\alpha \frac { 1 }{ 2m } \sum _{ i=1 }^{ m }{ { 2(W{ x }^{ i }-y^{ i } })x^{ i } }  $$$$ W:=W-\alpha \frac { 1 }{ m } \sum _{ i=1 }^{ m }{ { (W{ x }^{ i }-y^{ i } })x^{ i } }  $$
Gradient descent algorithm
$$ W:=W-\alpha \frac { 1 }{ m } \sum _{ i=1 }^{ m }{ { (W{ x }^{ i }-y^{ i } })x^{ i } }  $$
Convex function


Gradient descent algorithm을 사용하려면, 비용함수 cost(W,b)가 Convex function 이어야 한다

Gradient descent 구현
In [5]:
import tensorflow as tf

x_data = [1., 2., 3., 4.]
y_data = [1., 3., 5., 7.]

W = tf.Variable(tf.random_uniform([1], -1000., 1000.))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = W * X
cost = tf.reduce_mean(tf.square(hypothesis - Y))
mean = tf.reduce_mean(tf.multiply(tf.multiply(W, X) - Y, X))
descent = W - tf.multiply(0.01, mean)

w update
update  = W.assign(descent) 

init = tf.initializers.global_variables()

sess = tf.Session()
sess.run(init)

for step in range(2000):
    uResult = sess.run(update, feed_dict={X: x_data, Y: y_data}) # update W
    cResult = sess.run(cost, feed_dict={X: x_data, Y: y_data})
    wResult = sess.run(W)
    mResult = sess.run(mean, feed_dict={X: x_data, Y: y_data})
    if step % 100 == 0:
        print('%5d | %.15f | %.15f | %.15f | %.15f' %(step, mResult, cResult, wResult, uResult))
        
print('-' * 50)
print(sess.run(hypothesis, feed_dict={X: 5.0}))
print(sess.run(hypothesis, feed_dict={X: 2.5}))

sess.close()
    0 | 2820.839355468750000 | 1060951.500000000000000 | 377.778594970703125 | 377.778594970703125
  100 | 1.160249233245850 | 0.346157103776932 | 1.821366548538208 | 1.821366548538208
  200 | 0.000478029251099 | 0.166666701436043 | 1.666730403900146 | 1.666730403900146
  300 | 0.000005960464478 | 0.166666656732559 | 1.666667461395264 | 1.666667461395264
  400 | 0.000005960464478 | 0.166666656732559 | 1.666667461395264 | 1.666667461395264
  500 | 0.000005960464478 | 0.166666656732559 | 1.666667461395264 | 1.666667461395264
  600 | 0.000005960464478 | 0.166666656732559 | 1.666667461395264 | 1.666667461395264
  700 | 0.000005960464478 | 0.166666656732559 | 1.666667461395264 | 1.666667461395264
  800 | 0.000005960464478 | 0.166666656732559 | 1.666667461395264 | 1.666667461395264
  900 | 0.000005960464478 | 0.166666656732559 | 1.666667461395264 | 1.666667461395264
 1000 | 0.000005960464478 | 0.166666656732559 | 1.666667461395264 | 1.666667461395264
 1100 | 0.000005960464478 | 0.166666656732559 | 1.666667461395264 | 1.666667461395264
 1200 | 0.000005960464478 | 0.166666656732559 | 1.666667461395264 | 1.666667461395264
 1300 | 0.000005960464478 | 0.166666656732559 | 1.666667461395264 | 1.666667461395264
 1400 | 0.000005960464478 | 0.166666656732559 | 1.666667461395264 | 1.666667461395264
 1500 | 0.000005960464478 | 0.166666656732559 | 1.666667461395264 | 1.666667461395264
 1600 | 0.000005960464478 | 0.166666656732559 | 1.666667461395264 | 1.666667461395264
 1700 | 0.000005960464478 | 0.166666656732559 | 1.666667461395264 | 1.666667461395264
 1800 | 0.000005960464478 | 0.166666656732559 | 1.666667461395264 | 1.666667461395264
 1900 | 0.000005960464478 | 0.166666656732559 | 1.666667461395264 | 1.666667461395264

[8.333338]
[4.166669]
Liner regression Summary
1) Hypothesis
$$ H(x) = Wx + b $$
2) Cost function
$$ cost(W)=\frac { 1 }{ m } \sum _{i=1}^{m}{ { (W{ x }^{ i }-y^{ i } })^{ 2 } }  $$
3) Gradient descent
$$ W := W-\alpha \frac { \partial  }{ \partial W } cost(W) $$
Multi-variable linear regression
Predicting exam score - regression using three inputs (x1, x2, x3)

x1 (quiz 1)	x2 (quiz 2)	x3 (mid 1)	Y (final)
73	80	75	152
93	88	93	185
89	91	90	180
96	98	100	196
73	66	70	142
Test Scores for General Psychology 

Matrix multiplication
dot product(=scalar product, 내적)



Multi-feature regression
Hypothesis
$$ H(x) = w x + b $$$$ H(x_1, x_2, x_3) = w_1 x_1 + w_2 x_2 + w_3 x_3 + b $$
Hypothesis using matrix
$$ H(x_1, x_2, x_3) = \underline{w_1 x_1 + w_2 x_2 + w_3 x_3} + b $$$$ w_1 x_1 + w_2 x_2 + w_3 x_3 $$
$$ \begin{pmatrix} w_{ 1 } &amp; w_{ 2 } &amp; w_{ 3 } \end{pmatrix}\cdot \begin{pmatrix} x_{ 1 } \\ x_{ 2 } \\ x_{ 3 } \end{pmatrix} $$$$ WX $$
(W, X 는 matrix)

Hypothesis without b
$$ H(x_1, x_2, x_3) = w_1 x_1 + w_2 x_2 + w_3 x_3 + b$$$$ = b + w_1 x_1 + w_2 x_2 + w_3 x_3 $$$$ = \begin{pmatrix} b &amp; x_{ 1 } &amp; x_{ 2 } &amp; x_{ 3 } \end{pmatrix}\cdot \begin{pmatrix} 1 \\ w_{ 1 } \\ w_{ 2 } \\ w_{ 3 } \end{pmatrix} $$$$ = XW $$
Hypothesis using matrix
Many x instances
$$ \begin{pmatrix} x_{ 11 } &amp; x_{ 12 } &amp; x_{ 13 } \\ x_{ 21 } &amp; x_{ 22 } &amp; x_{ 23 } \\ x_{ 31 } &amp; x_{ 32 } &amp; x_{ 33 }\\ x_{ 41 } &amp; x_{ 42 } &amp; x_{ 43 }\\ x_{ 51 } &amp; x_{ 52 } &amp; x_{ 53 }\end{pmatrix} \cdot \begin{pmatrix} w_{ 1 } \\ w_{ 2 } \\ w_{ 3 } \end{pmatrix}=\begin{pmatrix} x_{ 11 }w_{ 1 }+x_{ 12 }w_{ 2 }+x_{ 13 }w_{ 3 } \\ x_{ 21 }w_{ 1 }+x_{ 22 }w_{ 2 }+x_{ 23 }w_{ 3 }\\ x_{ 31 }w_{ 1 }+x_{ 32 }w_{ 2 }+x_{ 33 }w_{ 3 } \\ x_{ 41 }w_{ 1 }+x_{ 42 }w_{ 2 }+x_{ 43 }w_{ 3 } \\ x_{ 51 }w_{ 1 }+x_{ 52 }w_{ 2 }+x_{ 53 }w_{ 3 } \end{pmatrix} $$$$ [5, 3] \cdot [3, 1] = [5, 1] $$$$ H(X) = XW $$
5는 데이터(instance)의 수, 3은 변수(feature)의 수, 1은 결과

Hypothesis using matrix (n output)
$$ [n, 3] \cdot [?, ?] = [n, 2] $$$$ H(X) = XW $$
n은 데이터(instance)의 개수, 2는 결과 값의 개수로 주어진다.
이때, W [?, ?] ⇒ [3, 2]
WX vs XW
Theory (Lecture) :
$$ H(x) = Wx + b  $$

TensorFlow (Implementation) :
$$ H(X) = XW $$
Simple Example (2 variables)
x1	x2	y
1	0	1
0	2	2
3	0	3
0	4	4
5	0	5
In [6]:
import tensorflow as tf

x1_data = [1, 0, 3, 0, 5]
x2_data = [0, 2, 0, 4, 0]
y_data  = [1, 2, 3, 4, 5]

W1 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b  = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

(hypothesis = W * X + b)
hypothesis = W1 * x1_data + W2 * x2_data + b

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

rate = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(rate)
train = optimizer.minimize(cost)

init = tf.initializers.global_variables()

sess = tf.Session()
sess.run(init)

for step in range(1,2001):
    sess.run(train)
    if step % 100 == 0:
        print("%-5d | %.15f | %f | %f | %f" % (step, sess.run(cost), sess.run(W1), sess.run(W2), sess.run(b)))

sess.close()
100   | 0.000457131478470 | 0.986676 | 0.984191 | 0.050686
200   | 0.000000930154215 | 0.999399 | 0.999287 | 0.002286
300   | 0.000000001895285 | 0.999973 | 0.999968 | 0.000103
400   | 0.000000000003899 | 0.999999 | 0.999999 | 0.000005
500   | 0.000000000000026 | 1.000000 | 1.000000 | 0.000000
600   | 0.000000000000048 | 1.000000 | 1.000000 | 0.000000
700   | 0.000000000000048 | 1.000000 | 1.000000 | 0.000000
800   | 0.000000000000048 | 1.000000 | 1.000000 | 0.000000
900   | 0.000000000000048 | 1.000000 | 1.000000 | 0.000000
1000  | 0.000000000000048 | 1.000000 | 1.000000 | 0.000000
1100  | 0.000000000000048 | 1.000000 | 1.000000 | 0.000000
1200  | 0.000000000000048 | 1.000000 | 1.000000 | 0.000000
1300  | 0.000000000000048 | 1.000000 | 1.000000 | 0.000000
1400  | 0.000000000000048 | 1.000000 | 1.000000 | 0.000000
1500  | 0.000000000000048 | 1.000000 | 1.000000 | 0.000000
1600  | 0.000000000000048 | 1.000000 | 1.000000 | 0.000000
1700  | 0.000000000000048 | 1.000000 | 1.000000 | 0.000000
1800  | 0.000000000000048 | 1.000000 | 1.000000 | 0.000000
1900  | 0.000000000000048 | 1.000000 | 1.000000 | 0.000000
2000  | 0.000000000000048 | 1.000000 | 1.000000 | 0.000000
Simple Example (2 variables)
In [7]:
import tensorflow as tf

x_data = [
    [1., 0., 3., 0., 5.],
    [0., 2., 0., 4., 0.]
]
y_data  = [1, 2, 3, 4, 5]

W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

hypothesis = tf.matmul(W, x_data) + b     # [1, 2] * [2, 5] = [1, 5]

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

rate = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(rate)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.initializers.global_variables())

for step in range(1,2001):
    sess.run(train)
    if step % 100 == 0:
        print("%-5d | %.15f | %s | %f " % (step, sess.run(cost), sess.run(W), sess.run(b)))
100   | 0.000288817245746 | [[0.9894093  0.98743427]] | 0.040288 
200   | 0.000000587746285 | [[0.99952227 0.99943316]] | 0.001817 
300   | 0.000000001193442 | [[0.9999784 0.9999744]] | 0.000082 
400   | 0.000000000002478 | [[0.9999991  0.99999887]] | 0.000004 
500   | 0.000000000000080 | [[1.         0.99999994]] | 0.000000 
600   | 0.000000000000014 | [[1.         0.99999994]] | 0.000000 
700   | 0.000000000000014 | [[1.         0.99999994]] | 0.000000 
800   | 0.000000000000014 | [[1.         0.99999994]] | 0.000000 
900   | 0.000000000000014 | [[1.         0.99999994]] | 0.000000 
1000  | 0.000000000000014 | [[1.         0.99999994]] | 0.000000 
1100  | 0.000000000000014 | [[1.         0.99999994]] | 0.000000 
1200  | 0.000000000000014 | [[1.         0.99999994]] | 0.000000 
1300  | 0.000000000000014 | [[1.         0.99999994]] | 0.000000 
1400  | 0.000000000000014 | [[1.         0.99999994]] | 0.000000 
1500  | 0.000000000000014 | [[1.         0.99999994]] | 0.000000 
1600  | 0.000000000000014 | [[1.         0.99999994]] | 0.000000 
1700  | 0.000000000000014 | [[1.         0.99999994]] | 0.000000 
1800  | 0.000000000000014 | [[1.         0.99999994]] | 0.000000 
1900  | 0.000000000000014 | [[1.         0.99999994]] | 0.000000 
2000  | 0.000000000000014 | [[1.         0.99999994]] | 0.000000 
Hypothesis without b
In [8]:
import tensorflow as tf

(앞의 코드에서 bias(b)를 행렬에 추가)
(갯수가 같아야 하므로 b를 리스트로 처리)

x_data = [
    [1., 1., 1., 1., 1.], 
    [1., 0., 3., 0., 5.], 
    [0., 2., 0., 4., 0.]
]
y_data  = [1, 2, 3, 4, 5]

W = tf.Variable(tf.random_uniform([1, 3], -1.0, 1.0)) # [1, 3]으로 변경하고, b 삭제
hypothesis = tf.matmul(W, x_data) # b가 없다

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

rate = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(rate)
train = optimizer.minimize(cost)

init = tf.initializers.global_variables()

sess = tf.Session()
sess.run(init)

for step in range(1,1001):
    sess.run(train)
    if step % 50 == 0:
        # without b
        print("%-5d | %.15f | %s | %f " % (step, sess.run(cost), sess.run(W), sess.run(b)))

sess.close()
50    | 0.005488837137818 | [[0.1756326 0.9538311 0.945221 ]] | -0.119597 
100   | 0.000247604621109 | [[0.03730303 0.9901941  0.98836535]] | -0.119597 
150   | 0.000011169533536 | [[0.00792286 0.99791735 0.9975289 ]] | -0.119597 
200   | 0.000000503816011 | [[0.00168275 0.9995577  0.9994751 ]] | -0.119597 
250   | 0.000000022737765 | [[3.5738791e-04 9.9990606e-01 9.9988854e-01]] | -0.119597 
300   | 0.000000001024549 | [[7.5920412e-05 9.9998003e-01 9.9997634e-01]] | -0.119597 
350   | 0.000000000045978 | [[1.6077338e-05 9.9999577e-01 9.9999499e-01]] | -0.119597 
400   | 0.000000000002004 | [[3.403004e-06 9.999991e-01 9.999989e-01]] | -0.119597 
450   | 0.000000000000114 | [[7.5655805e-07 9.9999976e-01 9.9999976e-01]] | -0.119597 
500   | 0.000000000000003 | [[1.8435321e-07 9.9999994e-01 9.9999994e-01]] | -0.119597 
550   | 0.000000000000000 | [[1.1282769e-07 9.9999994e-01 1.0000000e+00]] | -0.119597 
600   | 0.000000000000000 | [[1.1282769e-07 9.9999994e-01 1.0000000e+00]] | -0.119597 
650   | 0.000000000000000 | [[1.1282769e-07 9.9999994e-01 1.0000000e+00]] | -0.119597 
700   | 0.000000000000000 | [[1.1282769e-07 9.9999994e-01 1.0000000e+00]] | -0.119597 
750   | 0.000000000000000 | [[1.1282769e-07 9.9999994e-01 1.0000000e+00]] | -0.119597 
800   | 0.000000000000000 | [[1.1282769e-07 9.9999994e-01 1.0000000e+00]] | -0.119597 
850   | 0.000000000000000 | [[1.1282769e-07 9.9999994e-01 1.0000000e+00]] | -0.119597 
900   | 0.000000000000000 | [[1.1282769e-07 9.9999994e-01 1.0000000e+00]] | -0.119597 
950   | 0.000000000000000 | [[1.1282769e-07 9.9999994e-01 1.0000000e+00]] | -0.119597 
1000  | 0.000000000000000 | [[1.1282769e-07 9.9999994e-01 1.0000000e+00]] | -0.119597 
Multi-variable linear regression
In [9]:
 (Multi-variable linear regression)
import numpy as np
import tensorflow as tf

tf.set_random_seed(0)  # for reproducibility

data = np.array([
    [73,80,75,152],
    [93,88,93,185],
    [89,91,90,180],
    [96,98,100,196],
    [73,66,70,142]
])

x1_data = data[:,0]
x2_data = data[:,1]
x3_data = data[:,2]
y_data = data[:,3]

(placeholders for a tensor that will be always fed)
x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)

Y = tf.placeholder(tf.float32)
w1 = tf.Variable(tf.random_normal([1]), name='weight1')
w2 = tf.Variable(tf.random_normal([1]), name='weight2')
w3 = tf.Variable(tf.random_normal([1]), name='weight3')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b

(cost/loss function)
cost = tf.reduce_mean(tf.square(hypothesis - Y))

(Minimize cost)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.initializers.global_variables())

for step in range(4001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                          feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, Y: y_data})
    if step % 100 == 0:
        print('%4d: Cose=%8.4f' % (step, cost_val), "Prediction: ", hy_val)
   0: Cose=2667.6282 Prediction:  [105.34478  129.50781  125.94277  139.99611   97.017456]
 100: Cose=  3.8105 Prediction:  [151.11192 184.57016 180.1694  199.04204 139.03413]
 200: Cose=  3.7618 Prediction:  [151.08475 184.58958 180.16211 199.02852 139.06659]
 300: Cose=  3.7151 Prediction:  [151.05843 184.60847 180.15512 199.01526 139.09831]
 400: Cose=  3.6701 Prediction:  [151.03287 184.62682 180.14833 199.00217 139.12927]
 500: Cose=  3.6269 Prediction:  [151.0081  184.64462 180.1418  198.98927 139.15953]
 600: Cose=  3.5852 Prediction:  [150.9841  184.66193 180.1355  198.97658 139.1891 ]
 700: Cose=  3.5450 Prediction:  [150.96082 184.6787  180.1294  198.96404 139.21799]
 800: Cose=  3.5063 Prediction:  [150.93826 184.695   180.12355 198.9517  139.24623]
 900: Cose=  3.4689 Prediction:  [150.91638 184.7108  180.11787 198.9395  139.27379]
1000: Cose=  3.4329 Prediction:  [150.89517 184.72614 180.1124  198.92747 139.30074]
1100: Cose=  3.3981 Prediction:  [150.87462 184.74104 180.10715 198.91562 139.32707]
1200: Cose=  3.3644 Prediction:  [150.85472 184.7555  180.10208 198.90393 139.35283]
1300: Cose=  3.3319 Prediction:  [150.83545 184.76955 180.0972  198.89241 139.378  ]
1400: Cose=  3.3004 Prediction:  [150.81677 184.78314 180.0925  198.88101 139.40262]
1500: Cose=  3.2700 Prediction:  [150.79869 184.79634 180.088   198.86978 139.42668]
1600: Cose=  3.2405 Prediction:  [150.78116 184.80917 180.08365 198.85869 139.4502 ]
1700: Cose=  3.2119 Prediction:  [150.76419 184.8216  180.07947 198.84772 139.4732 ]
1800: Cose=  3.1842 Prediction:  [150.74779 184.83365 180.07547 198.83693 139.49571]
1900: Cose=  3.1573 Prediction:  [150.73192 184.84537 180.07162 198.82626 139.51773]
2000: Cose=  3.1312 Prediction:  [150.71652 184.85667 180.0679  198.81569 139.53925]
2100: Cose=  3.1058 Prediction:  [150.70163 184.86768 180.06435 198.80527 139.5603 ]
2200: Cose=  3.0811 Prediction:  [150.68726 184.87836 180.06096 198.79497 139.58092]
2300: Cose=  3.0572 Prediction:  [150.67331 184.88867 180.05768 198.78479 139.60106]
2400: Cose=  3.0338 Prediction:  [150.65987 184.8987  180.05458 198.77477 139.62082]
2500: Cose=  3.0111 Prediction:  [150.64684 184.90839 180.05157 198.7648  139.6401 ]
2600: Cose=  2.9889 Prediction:  [150.63426 184.91782 180.04872 198.75499 139.65901]
2700: Cose=  2.9673 Prediction:  [150.6221  184.92693 180.04599 198.74527 139.6775 ]
2800: Cose=  2.9462 Prediction:  [150.61037 184.93578 180.0434  198.73569 139.69563]
2900: Cose=  2.9256 Prediction:  [150.59901 184.94432 180.04091 198.72615 139.71335]
3000: Cose=  2.9055 Prediction:  [150.58806 184.9526  180.03854 198.71677 139.73073]
3100: Cose=  2.8859 Prediction:  [150.57747 184.96065 180.03629 198.70746 139.74773]
3200: Cose=  2.8667 Prediction:  [150.56726 184.96841 180.03413 198.69826 139.76437]
3300: Cose=  2.8479 Prediction:  [150.5574  184.97594 180.03209 198.68916 139.7807 ]
3400: Cose=  2.8295 Prediction:  [150.5479  184.98323 180.03017 198.68015 139.79669]
3500: Cose=  2.8115 Prediction:  [150.53874 184.99028 180.02832 198.67123 139.81235]
3600: Cose=  2.7938 Prediction:  [150.5299  184.9971  180.0266  198.66241 139.8277 ]
3700: Cose=  2.7765 Prediction:  [150.5214  185.00372 180.02498 198.65369 139.84274]
3800: Cose=  2.7596 Prediction:  [150.5132  185.0101  180.02342 198.64503 139.8575 ]
3900: Cose=  2.7429 Prediction:  [150.50528 185.01628 180.02197 198.63647 139.87195]
4000: Cose=  2.7265 Prediction:  [150.49767 185.02223 180.02058 198.62796 139.8861 ]

Lab 05 Logistic Classification(Regression) - Session
Logistic Classfication은 True or False와 같은 Binary나 복수개의 다항 분류에 쓰임 (Bernoulli Distribution)
기본 Library 선언 및 Tensorflow 버전 확인
In [1]:
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import tensorflow as tf

tf.set_random_seed(777)  # for reproducibility

print(tf.__version__)
/opt/conda/lib/python3.6/site-packages/h5py/__init__.py:34: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
1.12.0
x_data가 2차원 배열이기에 2차원 공간에 표현하여 x1과 x2를 기준으로 y_data 0과 1로 구분하는 예제
Logistic Classification 통해 보라색과 노란색 y_data(Label)을 구분
Test 데이터는 붉은색의 위치와 같이 추론시 1의 값을 가지게 됨
In [2]:
x_train = [[1, 2],
          [2, 3],
          [3, 1],
          [4, 3],
          [5, 3],
          [6, 2]]
y_train = [[0],
          [0],
          [0],
          [1],
          [1],
          [1]]

x_test = [[5,2]]
y_test = [[1]]


x1 = [x[0] for x in x_train]
x2 = [x[1] for x in x_train]

colors = [int(y[0] % 3) for y in y_train]
plt.scatter(x1,x2, c=colors , marker='^')
plt.scatter(x_test[0][0],x_test[0][1], c="red")

plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

Tensorflow Session
위 Data를 기준으로 가설의 검증을 통해 Logistic Classification 모델을 만듦
Tensorflow data API를 통해 학습시킬 값들을 담는다 (Batch Size는 한번에 학습시킬 Size로 정한다)
features,labels는 실재 학습에 쓰일 Data (연산을 위해 Type를 맞춰준다)
In [3]:
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(len(x_train)).repeat()

iter = dataset.make_initializable_iterator()
features, labels = iter.get_next()

features = tf.cast(features, tf.float32)
labels = tf.cast(labels, tf.float32)
위 Data를 기준으로 가설의 검증을 통해 Logistic Classification 모델을 만듦
W와 b은 학습을 통해 생성되는 모델에 쓰이는 Wegith와 Bias (초기값을 variable : 0이나 Random값으로 가능 tf.random_normal([2, 1]) )
In [4]:
W = tf.Variable(tf.zeros([2,1]), name='weight')
b = tf.Variable(tf.zeros([1]), name='bias')
Sigmoid 함수를 가설로 선언
Sigmoid는 아래 그래프와 같이 0과 1의 값만을 리턴 tf.sigmoid(tf.matmul(X, W) + b)와 같다
$$
\begin{align}
sigmoid(x) &amp; = \frac{1}{1+e^{-x}}  \\\\\
\end{align}
$$
sigmoid

In [5]:
hypothesis  = tf.div(1., 1. + tf.exp(tf.matmul(features, W) + b))
가설을 검증할 Cost 함수를 정의
$$
\begin{align}
cost(h(x),y) &amp; = −log(h(x))  &amp;  if  &amp;  y=1 \\\\\
cost(h(x),y) &amp; = -log(1−h(x))  &amp;  if  &amp;  y=0
\end{align}
$$
위 두수식을 합치면 아래과 같다 $$
\begin{align}
cost(h(x),y) &amp; = −y log(h(x))−(1−y)log(1−h(x))
\end{align}
$$
In [6]:
cost = -tf.reduce_mean(labels * tf.log(hypothesis) + (1 - labels) * tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
추론한 값은 0.5를 기준(Sigmoid 그래프 참조)로 0과 1의 값을 리턴
Sigmoid 함수를 통해 예측값이 0.5보다 크면 1을 반환하고 0.5보다 작으면 0으로 반환
In [7]:
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
가설을 통해 실재 값과 비교한 정확도를 측정
In [8]:
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, labels), dtype=tf.int32))
Tensorflow를 통한 실행을 위해 Session를 선언
위의 Data를 Cost함수를 통해 학습시킨 후 모델을 생성
새로운 Data를 통한 검증 수행 [5,2]의 Data로 테스트 수행 (그래프상 1이 나와야 함)
In [9]:
EPOCHS = 10001

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(EPOCHS):
        sess.run(iter.initializer)
        _, loss_value = sess.run([train, cost])
        if step % 1000 == 0:
            print("Iter: {}, Loss: {:.4f}".format(step, loss_value))
    h, c, a = sess.run([hypothesis, predicted, accuracy])
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)
    print("\nTest Data : {}, Predict : {}".format(x_test, sess.run(predicted, feed_dict={features: x_test})))
Iter: 0, Loss: 0.6931
Iter: 1000, Loss: 0.4145
Iter: 2000, Loss: 0.3496
Iter: 3000, Loss: 0.3014
Iter: 4000, Loss: 0.2636
Iter: 5000, Loss: 0.2336
Iter: 6000, Loss: 0.2094
Iter: 7000, Loss: 0.1896
Iter: 8000, Loss: 0.1731
Iter: 9000, Loss: 0.1592
Iter: 10000, Loss: 0.1474

Hypothesis:  [[0.02987642]
 [0.1576593 ]
 [0.30070737]
 [0.78328896]
 [0.9407705 ]
 [0.98057085]] 
Correct (Y):  [[0.]
 [0.]
 [0.]
 [1.]
 [1.]
 [1.]] 
Accuracy:  1

Test Data : [[5, 2]], Predict : [[1.]]
In [ ]:
