# PIL module is used to extract 
# pixels of image and modify it 
from PIL import Image 
my_dict = {}
# Convert encoding data into 8-bit binary 
# form using ASCII value of characters 
def genData(data): 
        
        # list of binary codes 
        # of given data 
        newd = [] 
        
        for i in data: 
            newd.append(format(ord(i), '08b')) #convert to ascii then to binary list
        #print(newd)
        return newd 
        
# Pixels are modified according to the 
# 8-bit binary data and finally returned 
def modPix(pix, data): 
    
    datalist = genData(data) #binary list
    #print("datalist",datalist)
    lendata = len(datalist) 
    #print("length",lendata)
    imdata = iter(pix) #The iter() method returns iterator object for the given object that loops through each element in the object.
    for i in range(lendata): 
        
        # Extracting 3 pixels at a time 
        #p1 = imdata.__next__()[:3]
        #print(p1)
        pix = [value for value in imdata.__next__()[:3] +
                                imdata.__next__()[:3] +
                                imdata.__next__()[:3]] 
        #print("pix",pix)                           
        # Pixel value should be made 
        # odd for 1 and even for 0 
        for j in range(0, 8): 
            #print("öld",pix[j])
            if (datalist[i][j]=='0') and (pix[j]% 2 != 0): #prints new value with -1 only if datalist is 0 and pix is odd number
                
                if (pix[j]% 2 != 0): 
                    pix[j] -= 1
                    #print(pix[j])
            elif (datalist[i][j] == '1') and (pix[j] % 2 == 0): #datalist has 1 and pix is even
                pix[j] -= 1
                #print(pix[j])       
        # Eigh^th pixel of every set tells 
        # whether to stop ot read further. 
        # 0 means keep reading; 1 means the 
        # message is over. 
        if (i == lendata - 1): 
            #print("old",pix[-1])
            if (pix[-1] % 2 == 0): 
                pix[-1] -= 1
                #print("even",pix[-1])

        else: 
            if (pix[-1] % 2 != 0): 
                pix[-1] -= 1
                #print("odd",pix[-1])
        pix = tuple(pix) #list creation
        #print("tuple",pix)
        yield pix[0:3]
        yield pix[3:6] 
        yield pix[6:9] 

def encode_enc(newimg, data): 
    w = newimg.size[0] #height width
    #print(newimg.size)
    #print(w)
    (x, y) = (0, 0) 
    for pixel in modPix(newimg.getdata(), data): #getdata-ordinary sequence
        
        # Putting modified pixels in the new image 
        newimg.putpixel((x, y), pixel) 
        if (x == w - 1): 
            x = 0
            y += 1
        else: 
            x += 1
            
# Encode data into image 
def encode(): 
    img = input("Enter image name(with extension): ") 
    image = Image.open(img, 'r') 
    
    #data = input("Enter data to be encoded : ") 
    with open('dec.txt') as fileobj:
        for line in fileobj:
            key, value = line.split(":")
            my_dict[key] = value

    key = my_dict['keyen']
    data = my_dict['seq']
    if (len(data) == 0): 
        raise ValueError('Data is empty') 
        
    newimg = image.copy() 
    encode_enc(newimg, data) 
    
    new_img_name = input("Enter the name of new image(with extension): ") 
    newimg.save(new_img_name, str(new_img_name.split(".")[1].upper())) 

# Decode the data in the image 
def decode(): 
    img = input("Enter image name(with extension) :") 
    image = Image.open(img, 'r') 
    
    data = '' 
    imgdata = iter(image.getdata()) 
    
    while (True): 
        pixels = [value for value in imgdata.__next__()[:3] +
                                imgdata.__next__()[:3] +
                                imgdata.__next__()[:3]] 
        # string of binary data 
        binstr = '' 
        
        for i in pixels[:8]: 
            if (i % 2 == 0): 
                binstr += '0'
            else: 
                binstr += '1'
                
        data += chr(int(binstr, 2)) 
        if (pixels[-1] % 2 != 0): 
            return data 
            
# Main Function      
def main(): 
    a = int(input("1. Encode\n 2. Decode\n")) 
    if (a == 1): 
        encode() 
        
    elif (a == 2): 
        print("Decoded word- " + decode()) 
    else: 
        raise Exception("Enter correct input") 
        
# Driver Code 
if __name__ == '__main__' : 
    
    # Calling main function 
    main() 
