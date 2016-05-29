# from svmutil import *
# y, x = svm_read_problem('./example/w1a')
# prob = svm_problem(y, x)
# param = svm_parameter('-s 0 -t 2 -c 10 -g 0.1')
# m = svm_train(prob, param)
# svm_save_model('./example/w1a_model', m)
#
# model = svm_load_model('./example/w1a_model')
# y1, x1 = svm_read_problem('./example/w1a.t')
# p_labels, p_acc, p_vals = svm_predict(y1, x1, model)
# print '>>>>', y1
# print '>>>>', p_labels


# from svmutil import *
# y, x = svm_read_problem('lbp_train2.txt')
# prob = svm_problem(y, x)
# param = svm_parameter('-s 0 -t 2 -c 10 -g 0.1')
# m = svm_train(prob, param)
# svm_save_model('model.txt', m)
#
# model = svm_load_model('model.txt')
# y1, x1 = svm_read_problem('lbp_test2.txt')
# p_labels, p_acc, p_vals = svm_predict(y1, x1, model)
# print '>>>>', y1
# print '>>>>', p_labels


from svmutil import *
fn = 'h_scale'
step = 5
acc = 0.0
for start in range(step):
    y, x = svm_read_problem('./train/'+fn+'_train_'+str(step)+'_'+str(start))
    prob = svm_problem(y, x)
    param = svm_parameter('-s 0 -t 0')
    m = svm_train(prob, param)
    svm_save_model('./model/'+fn+'_model_'+str(step)+'_'+str(start), m)

    model = svm_load_model('./model/'+fn+'_model_'+str(step)+'_'+str(start))
    y1, x1 = svm_read_problem('./test/'+fn+'_test_'+str(step)+'_'+str(start))
    p_labels, p_acc, p_vals = svm_predict(y1, x1, model)
    print 'problem:', y1
    print 'result:', p_labels
    acc += p_acc[0]
acc /= step
print 'average_acc:', acc


# from svmutil import *
# import numpy
# import matplotlib.pyplot as plt
# labels, data = svm_read_problem('h_test.txt')
# for i in range(300, 310):
#     vals = dict(data[i]).values()
#     print vals
#     x = numpy.arange(len(vals))
#     fig_h = plt.subplot(111)
#     fig_h.bar(x, vals, alpha=.5, color='r')
#     plt.show(fig_h)



# options:
# -s svm_type : set type of SVM (default 0)
# 	0 -- C-SVC
# 	1 -- nu-SVC
# 	2 -- one-class SVM
# 	3 -- epsilon-SVR
# 	4 -- nu-SVR
# -t kernel_type : set type of kernel function (default 2)
# 	0 -- linear: u'*v
# 	1 -- polynomial: (gamma*u'*v + coef0)^degree
# 	2 -- radial basis function: exp(-gamma*|u-v|^2)
# 	3 -- sigmoid: tanh(gamma*u'*v + coef0)
# -d degree : set degree in kernel function (default 3)
# -g gamma : set gamma in kernel function (default 1/num_features)
# -r coef0 : set coef0 in kernel function (default 0)
# -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
# -n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
# -p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
# -m cachesize : set cache memory size in MB (default 100)
# -e epsilon : set tolerance of termination criterion (default 0.001)
# -h shrinking: whether to use the shrinking heuristics, 0 or 1 (default 1)
# -b probability_estimates: whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
# -wi weight: set the parameter C of class i to weight*C, for C-SVC (default 1)
#
# The k in the -g option means the number of attributes in the input data.