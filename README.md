# ml from scratch
Simple AI Learns y = 2x

This is a super basic machine learning model I made that learns the pattern:

y = 2x

It uses something called linear regression, but basically it just keeps guessing and fixing itself until it gets the right answer.

How it works

The AI starts off dumb with random values:

weight → controls how strong x affects the result
bias → shifts the line up or down
learning rate → how fast it learns

Then it goes through the data over and over and keeps improving.

Training data
(1, 2)
(2, 4)
(3, 6)
(4, 8)

So it’s just learning the pattern: multiply x by 2.

What the AI does

For every number:

Makes a prediction:
prediction = weight * x + bias
Checks how wrong it is:
error = prediction - y
Fixes itself using that error

It repeats this like a million times until it gets really close.

Stop condition

If the error gets super small, it stops training early.

After training

You can type any number and it will predict the answer using what it learned.
