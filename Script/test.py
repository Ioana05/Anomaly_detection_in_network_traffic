s = "SKY_func4.exe/ribbit>"
nr = 0
if s[0] == s[1]:
    print("ignore")

if s[len(s)-1] == 'a' or s[len(s)-1] == 'e' or s[len(s)-1] == 'i' or s[len(s)-1] == 'o' or s[len(s)-1] == 'u':
    s = s + ".vowel"
for c in range(len(s)-1):
    if s[c] in '0123456789':
        s = s[:c] + s[c] + '.number'
        nr += 1
if nr >= 2:
    print("Prima litera trebuie sa fie mare")
