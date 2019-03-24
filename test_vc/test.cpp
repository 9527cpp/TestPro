#include <iostream>
using namespace std;

int getMax(int *a,int i){
	if(i==0)return a[0];
	int t=getMax(a,i-1);
	return t>a[i-1]?t:a[i-1];
}

int getSum(int *a,int i){
	if(i==0)return 0;
	return getSum(a,i-1)+a[i-1];	
}

float getAvg(int *a,int i){
	if(i==0)return a[0];
	return (float(getAvg(a,i-1)*(i-1)+a[i-1]))/i;
}

int main()
{
	int a[5]={2,5,9,1,8};
	cout<<getMax(a,5)<<endl;
	cout<<getSum(a,5)<<endl;
	cout<<getAvg(a,5)<<endl;
	return 0;
}