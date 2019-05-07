def solution(a):

    m = {}
    for i in range(len(a)):
        m[a[i]] = i 
        
    print(m)
    sorted_a = sorted(a, reverse = True)
    ret = 0
    
    for i in range(len(a)):
        if a[i] != sorted_a[i]:
            ret +=1
            
            ind_to_swap = m[ sorted_a[i] ]
            m[ a[i] ] = m[ sorted_a[i]]
            a[i],a[ind_to_swap] = sorted_a[i],a[i]
    return ret


a = [3,1,2,4]

asc=solution(list(a))
desc=solution([3,1,2,4])
print (desc)