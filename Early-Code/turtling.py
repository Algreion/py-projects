import turtle
import random
import sys
from math import pi

def dvd_setup(t, draw, color, xpos, ypos, hide, speed, xb, yb):
    t.pu()
    t.speed(0)
    t.goto(-xb,-yb)
    t.turtlesize(2)
    t.pencolor(color)
    t.pensize(5)
    t.pd()
    t.goto(-xb,yb)
    t.goto(xb,yb)
    t.goto(xb,-yb)
    t.goto(-xb,-yb)
    t.pu()
    t.goto(xpos,ypos)
    t.speed(speed)
    t.pensize(1)
    if draw:
        t.pd()
    if hide: t.hideturtle()

def dvd(t, angle=None,speed=5,draw=False,color=(0,0,0),xpos=None,ypos=None,hide = False):
    if angle == None:
        angle = int(random.randrange(30,60,1))
    if xpos == None or ypos == None:
        xpos,ypos = int(random.randrange(-75,75,1)),int(random.randrange(-75,75,1))
    xb, yb = 320, 180
    dvd_setup(t,draw,color,xpos,ypos,hide,speed,xb,yb)
    t.lt(angle)
    while True:
        x,y = t.pos()
        if abs(x) >= xb-2 and abs(y) >= yb-2:
            break
        if abs(x) >= xb-2:
            a = t.heading()
            t.speed(0)
            t.setheading(180-a)
            t.speed(speed)
        elif abs(y) >= yb-2:
            a = t.heading()
            t.speed(0)
            t.setheading(-a)
            t.speed(speed)
        t.fd(5)
    print("You hit a corner!")

def tree(t, l, n, a=30, setup=False):
    if setup:
        try:
            t.speed(0)
            t.pu()
            t.lt(90)
            t.goto(0,-300)
            t.pd()
            t.pencolor("brown")
            t.pensize(1.5)
            tree(t,l,n,a)
            t.hideturtle()
            turtle.done()
        except: return
        return

    if n == 1:
        t.pencolor("green")
        t.pensize(3)
    if n <= 0:
        return
    t.fd(l*(n+1))
    t.lt(a)
    tree(t,l,n-1)
    t.rt(2*a)
    tree(t,l,n-1)
    t.lt(a)
    t.bk(l*(n+1))
    if n == 1:
        t.pencolor("brown")
        t.pensize(1.5)

def koch(t, l, setup=False):
    if setup:
        try:
            t.hideturtle()
            t.pu()
            t.goto(-450,0)
            t.pd()
            t.speed(0)
            t.color(0.55,0.1,1)
            koch (t,900)
            turtle.done()
        except: print("Cancelled")
        return

    if l < 3:
        t.fd(l+1)
        return
    koch(t,l/3)
    t.lt(60)
    koch(t,l/3)
    t.rt(120)
    koch(t,l/3)
    t.lt(60)
    koch(t,l/3)

def snowflake(t, setup=False):
    if setup:
        t.hideturtle()
        t.pu()
        t.pd()
        t.speed(0)
        t.color(0,0.5,1)
        snowflake(t)
        turtle.done()
        return
    t.lt(90)
    koch(t,100)
    t.rt(120)
    koch(t,100)
    t.rt(120)
    koch(t,100)

def square(t,l):
    for _ in range(4):
        t.fd(l)
        t.lt(90)
def square_rep(t,n=20,l=15):
    for i in range(n):
        square(t,i*l)
    t.hideturtle()
def square_tilt(t,n=30,l=7,a=10):
    r = random.random()
    for i in range(n):
        t.pencolor(r,r,r)
        t.lt(a) if i else t.lt(0)
        square(t,i*l)
    t.hideturtle()
def circle(t,r,precision=100):
    c = 2*pi*r
    step = c/precision
    angle = 360/precision
    for _ in range(precision):
        t.fd(step)
        t.lt(angle)
def spiral(t,n=300,width=15):
    for i in range(n):
        t.fd(1+i/10)
        t.lt(width)
def leaf():
    for i in range(18):
        m = turtle.Turtle()
        m.speed(0)
        m.lt(10*i)
        spiral(m,60,5)
        m.hideturtle()

if __name__  == '__main__':
    USAGE = "Usage: python turtling.py [snowflake/tree/dvd/spiral/square_tilt/circle/square] (params)"
    if len(sys.argv) > 1:
        bob = turtle.Turtle()
        try:
            match sys.argv[1].lower():
                    case 'circle' | 'c':
                        r = int(sys.argv[2]) if len(sys.argv)>2 else 100
                        p = int(sys.argv[3]) if len(sys.argv)>3 else 100
                        circle(bob,r,p)
                        turtle.done()
                    case 'spiral':
                        n = int(sys.argv[2]) if len(sys.argv)>2 else 300
                        w = int(sys.argv[3]) if len(sys.argv)>3 else 15
                        spiral(bob, n, w)
                        turtle.done()
                    case 'square_tilt' | 'st' | 'sqti':
                        n = int(sys.argv[2]) if len(sys.argv)>2 else 30
                        l = int(sys.argv[3]) if len(sys.argv)>3 else 7
                        a = int(sys.argv[4]) if len(sys.argv)>4 else 10
                        square_tilt(bob, n, l, a)
                        turtle.done()
                    case 'square' | 'sq':
                        l = int(sys.argv[2]) if len(sys.argv)>2 else 100
                        square(bob, l)
                        turtle.done()
                    case 'snowflake' | 's' | 'snow':
                        s = int(sys.argv[2]) if len(sys.argv)>2 else 1
                        snowflake(bob, setup=s)
                        turtle.done()
                    case 'tree' | 't':
                        l = int(sys.argv[2]) if len(sys.argv)>2 else 20
                        n = int(sys.argv[3]) if len(sys.argv)>3 else 6
                        a = int(sys.argv[4]) if len(sys.argv)>4 else 30
                        s = int(sys.argv[5]) if len(sys.argv)>5 else 1
                        tree(bob, l, n, a, s)
                        turtle.done()
                    case 'dvd' | 'd':
                        a = int(sys.argv[2]) if len(sys.argv)>2 else None
                        if a == 0: a = None
                        s = int(sys.argv[3]) if len(sys.argv)>3 else 5
                        d = int(sys.argv[4]) if len(sys.argv)>4 else 0
                        c = ('red' if (x:=int(sys.argv[5]))==1 else 'green' if x==2 else 'blue' if x==3 else (0,0,0)) if len(sys.argv)>5 else (0,0,0)
                        hide = int(sys.argv[6]) if len(sys.argv)>6 else 0
                        dvd(bob,a,s,d,c,None,None,hide)
                        turtle.done()
                    case _:
                        print(USAGE)
                        snowflake(bob,True)
                        turtle.done()
        except:
            sys.exit(f"You just turtled!")
    else:
        print(USAGE)