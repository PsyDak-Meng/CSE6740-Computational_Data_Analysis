import numpy as np
import imageio
from matplotlib import pyplot as plt
import sys
import os
import time

def mykmeans(pixels, K):
    start=time.time()
    """
    Your goal of this assignment is implementing your own K-means.
​
    Input:
        pixels: data set. Each row contains one data point. For image
        dataset, it contains 3 columns, each column corresponding to Red,
        Green, and Blue component.
​
        K: the number of desired clusters. Too high value of K may result in
        empty cluster error. Then, you need to reduce it.
​
    Output:
        class: the class assignment of each data point in pixels. The
        assignment should be 0, 1, 2, etc. For K = 5, for example, each cell
        of class should be either 0, 1, 2, 3, or 4. The output should be a
        column vector with size(pixels, 1) elements.
​
        centroid: the location of K centroids in your result. With images,
        each centroid corresponds to the representative color of each
        cluster. The output should be a matrix with K rows and
        3 columns. The range of values should be [0, 255].
    """
    a1=pixels.shape[0]
    a2=pixels.shape[1]
    count=np.zeros((K,1))
    label=np.random.rand(pixels.shape[0],pixels.shape[1],1)
    while (0 in count):
        centroids=np.random.randint(0,256,size=(K,3))
        #iter_count=0
        count=np.zeros((K,1))
        p=pixels.reshape(a1*a2,3)
        for center in range(0,K):
            centroids[center]=p[int(center/K*a1*a2)]
        distance=np.zeros((K,a1,a2))
        cluster=np.zeros((a1,a2,4))
        cluster[:,:,0:3]=pixels
        while not np.allclose(label,np.reshape(cluster[:,:,3],(a1,a2,1)),atol=0):     
            #while not np.allclose(label,np.reshape(cluster[:,:,3],(a1,a2,1)),atol=0):
            i=0
            #iter_count+=1
            print(cluster[:,:,3],'1')
            np.copyto(label,np.reshape(cluster[:,:,3],(a1,a2,1)))
            for c in centroids:
                diff=pixels-c
                distance[i]=np.sum(diff**2,axis=2)**0.5
                i+=1
            distance_stack=np.dstack([distance[k] for k in range(0,K)])
            #print(distance_stack)
            #distance=distance_stack[:,:].min(axis=2)
            #print('distance:\n',distance.shape,distance)
            cluster[:,:,3]=np.argsort(distance_stack)[:,:,0]
            print(cluster[:,:,3],'2')
            for k in range(0,K):
                count[k,0]=cluster[cluster[:,:,3]==k].shape[0]
                centroids[k]+=np.sum(cluster[cluster[:,:,3]==k][:,0:3].astype('int32'),axis=0)
            centroids=centroids/count
        if (0 in count):
            print('There are empty clusters in K-means, should try a smaller K.\nFor a smaller K:' )
            print('\nK-means count:\n',count)
            #del(cluster,count,centroids,distance,distance_stack)
            K=input()
            K=int(K)
            #break
    #raise NotImplementedError
    #print('Total: ',iter_count,' runs.')
    #np.reshape(cluster[:,:,3],(68480,1))
    print('K-means count:\n',count)
    end=time.time()
    print('K-means runtime: ',end-start)
    return np.reshape(cluster[:,:,3].flatten(),(a1*a2,1)).astype('int32'),centroids

def Manhatten_dist(diff):
    return np.absolute(np.sum(diff))
def inf_dist(diff):
    return np.absolute(diff).max()

def mykmedoids(pixels, K):
    start=time.time()
    """
    Your goal of this assignment is implementing your own K-medoids.
    Please refer to the instructions carefully, and we encourage you to
    consult with other resources about this algorithm on the web.
​
    Input:
        pixels: data set. Each row contains one data point. For image
        dataset, it contains 3 columns, each column corresponding to Red,
        Green, and Blue component.
​
        K: the number of desired clusters. Too high value of K may result in
        empty cluster error. Then, you need to reduce it.
​
    Output:
        class: the class assignment of each data point in pixels. The
        assignment should be 0, 1, 2, etc. For K = 5, for example, each cell
        of class should be either 0, 1, 2, 3, or 4. The output should be a
        column vector with size(pixels, 1) elements.
​
        centroid: the location of K centroids in your result. With images,
        each centroid corresponds to the representative color of each
        cluster. The output should be a matrix with K rows and
        3 columns. The range of values should be [0, 255].
    """
    a1=pixels.shape[0]
    a2=pixels.shape[1]
    count=np.zeros((K,1))
    while (0 in count):
        centroids=np.random.randint(0,256,size=(K,3))
        p=pixels.reshape(a1*a2,3)
        for center in range(0,K):
            centroids[center]=p[int(center/K*a1*a2)]
        clusters=np.random.randint(0,K,size=(a1*a2,4))
        pixels=pixels.reshape(a1*a2,3)
        clusters[:,0:3]=pixels
        count=np.zeros((K,1))
        distance=np.array([float('inf') ,0])
        while distance[0]>distance[1]:          
            distance[0]=distance[1]
            distance[1]=0
            centroids_temp=np.copy(centroids)
            for i in clusters:
                temp=[]
                for c in centroids:
                    diff=Manhatten_dist(i[0:3]-c)
                    temp.append(diff)
                    distance[1]+=diff
                i[3]=temp.index(min(temp))
                #print(temp,min(temp))
                #print('/n',temp.index(min(temp)))
                count[temp.index(min(temp))]+=1
                centroids_temp[temp.index(min(temp))]+=i[0:3]
            centroids=centroids_temp/count
        if (0 in count):
            print('There are empty clusters in K-mediods, should try a smaller K.\nFor a smaller K:' )
            print('K-medoids count:\n',count)
            #del(cluster,count,centroids,distance,distance_stack)
            K=input()
            K=int(K)
            #break
            
    #raise NotImplementedError
    #print('Total: ',iter_count,' runs.')
    #np.reshape(cluster[:,:,3],(68480,1))
    print('K-medoids count:\n',count)
    print('/n',centroids)
    end=time.time()
    print('K-medoids runtime: ',end-start)
    return clusters[:,3].reshape(a1*a2,1),centroids
    #raise NotImplementedError

def main():
	if(len(sys.argv) < 2):
		print("Please supply an image file")
		return

	image_file_name = sys.argv[1]
	K = 5 if len(sys.argv) == 2 else int(sys.argv[2])
	print(image_file_name, K)
	im = np.asarray(imageio.imread(image_file_name))

	fig, axs = plt.subplots(1, 2)

	classes, centers = mykmedoids(im, K)
	print(classes, centers)
	new_im = np.asarray(centers[classes].reshape(im.shape), im.dtype)
	imageio.imwrite(os.path.basename(os.path.splitext(image_file_name)[0]) + '_converted_mykmedoids_' + str(K) + os.path.splitext(image_file_name)[1], new_im)
	axs[0].imshow(new_im)
	axs[0].set_title('K-medoids')

	classes, centers = mykmeans(im, K)
	print('K-means labels:\n',classes,'\nK-means centers:\n', centers)
	new_im = np.asarray(centers[classes].reshape(im.shape), im.dtype)
	imageio.imwrite(os.path.basename(os.path.splitext(image_file_name)[0]) + '_converted_mykmeans_' + str(K) + os.path.splitext(image_file_name)[1], new_im)
	axs[1].imshow(new_im)
	axs[1].set_title('K-means')

	plt.show()

if __name__ == '__main__':
	main()

#homework1.py football.bmp 50
#homework1.py pic5-2.jpg 16

