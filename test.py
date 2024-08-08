def test(*args, **kwargs):
    for key, value in kwargs.items():
        print("{0} = {1}".format(key, value))
    for arg in args:
        print("{0}".format(arg))

test('sup', a=1,)