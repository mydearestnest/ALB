# def p_out(f):#装饰器的基本形式
#     def p_out2(wd, wd2):#其中wd，wd2与p_try中的test，test2是直接对接的（参数对接）
#         print('fuck you bitch')
#         f(wd, wd2)
#     return p_out2
#
#
# @p_out
# def p_try(word1, word2):
#
#     print("I think you want to say something")
#     print(word1)
#     print(word2)
#     print("fuck you bitch again")
#
#
# test = "hey,fuck you son of bitch"
# test2 = "2"
# p_try(test, test2)

# 在Python中的代码中经常会见到这两个词 args 和 kwargs，前面通常还会加上一个或者两个星号。
# 其实这只是编程人员约定的变量名字，args 是 arguments 的缩写，表示位置参数；kwargs 是 keyword arguments 的缩写，表示关键字参数。
# 这其实就是 Python 中可变参数的两种形式，并且 *args 必须放在 **kwargs 的前面，因为位置参数在关键字参数的前面。
# 使用*args与*kwargs可传递任意变量，在变量未知的情况下很好用，将上述代码改写

def p_out(f):  # 装饰器的基本形式
    def p_out2(*args, **kwargs):  # 其中wd，wd2与p_try中的test，test2是直接对接的（参数对接）
        print('fuck you bitch')
        f(*args, **kwargs)
    return p_out2


@p_out
def p_try(word1, word2):

    print("I think you want to say something")
    print(word1)
    print(word2)
    print("fuck you bitch again")


test = "hey,fuck you son of bitch"
test2 = "2"
p_try(test, test2)
