fn = 'h_scale'
step = 5
for start in range(step):
    with open(fn, 'r') as f, \
            open('../train/'+fn+'_train_'+str(step)+'_'+str(start), 'w') as tr, \
            open('../test/'+fn+'_test_'+str(step)+'_'+str(start), 'w')as te:
        i = 0
        for line in f:
            if i % step == start:
                te.write(line)
            else:
                tr.write(line)
            i += 1
