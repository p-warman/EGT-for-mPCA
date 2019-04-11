import numpy as np
from matplotlib import pyplot as plt
import math
from ggplot import *


class BoneModel:
    
    rhoStart = np.array([0.001, 0.01, 0.0005, 0.0001])
    tmax = 800
    alpha = 0.005
    delta = 1.5
    gamma = 300
    epsilon = .03
    r = 0.1
    B_timescale = 1/2.5
    selection_strength = 0.05
    BStart = 1
    threshold = 0 
    
    s_all = np.zeros(tmax)
    s_strength = 0
    chemo_blocks = []
    
    rho_all = np.zeros((tmax,4))
    B_all = np.zeros(tmax)
    warray = np.zeros(4)
    fwarray = np.zeros((tmax, 4))


    def __init__(self):
        print(self.tmax)
        self.threshold = 0
        # print self.rho_all
    
    def ChangeTime(self, tnew):
        self.tmax = tnew
        self.ClearChemo()
        self.rho_all = np.zeros((tnew,4))
        self.B_all = np.zeros(tnew)
    
    def Homeostasis(self):
        #Homeostasis with initial bone perturbation
        self.rhoStart = [0.001, 0.01, 0.0, 0.0]
        self.BStart = 1
    
    def TumorIntroduction(self):
        #Tumor Introduction
        self.rhoStart = [0.001, 0.01, 0.0005, 0]
        self.BStart = 1.0
    
    def AllPlayers(self):
        #Tumor Introduction
        self.rhoStart = [0.001, 0.01, 0.0005, 0.0001]
        self.BStart = 1.0
        self.rho_all = np.zeros((self.tmax,4))

    
    def ClearRhoStart(self):
        self.rhoStart = np.array([0, 0, 0, 0])
        
        return self.rhoStart
    
    def AddRemodel(self,rho_b = 0.001,rho_c = 0.01):
        if np.sum(self.rhoStart) > 1 - (rho_b + rho_c):
            print('There is not enough room!')
            return self.rhoStart
        
        self.rhoStart = self.rhoStart + [rho_b,rho_c,0,0]
        return self.rhoStart
    
    def AddTumour(self,p_sen = 1, tum_size = 0.0005):
        if np.sum(self.rho_all[0,:]) > 1 - tum_size:
            print('There is not enough room!')
            return self.rho_all
        
        self.rhoStart = self.rhoStart + [0,0,p_sen*tum_size,(1 - p_sen)*tum_size]
        return self.rhoStart
    
    def ClearChemo(self):
        #Reset chemo to 0 for all time points
        self.s_all = np.zeros(self.tmax)
        self.chemo_blocks = []
        
        return self.s_all
    
    def AddChemoBlock(self,t_start,t_end,s_strength):
        #Create one block of chemo at specified time and strength
        self.s_all[t_start:t_end] = s_strength*np.ones(t_end - t_start)
        self.chemo_blocks.append((t_start,t_end))
        
        return self.s_all
    
    def MakeChemoScheme(self, t_start, t_length, t_break, n_blocks, s_strength):
        #Create chemo scheme for prescribed durations/regiments
        self.ClearChemo()
        
        for i in range(n_blocks):
            t_start_block = t_start + i*(t_length + t_break)
            self.AddChemoBlock(t_start_block, t_start_block + t_length, s_strength)
        
        return self.s_all

    def ChemoPlot(self):
        plt.plot(self.s_all)
        plt.ylim([0,1])
        plt.show()


    def CompFit(self,rho, B, s):
        # rho: 4x1 array that has [RhoB, RhoC, RhoS, RhoR]
        # B: single number for bone
        # s: current strength of therapy
        # function returns w: a 4 by 1 array that has [w_b, w_c, w_s, w_r]
        
        self.warray[0] = ((rho[1]/(rho[0] + rho[1]))*(1 - B)  + (rho[2] + rho[3])*self.delta)*(1-2*s*0.75)
        self.warray[1] = (rho[0]/(rho[0] + rho[1]))*(B - 1)*(1-2*s*0.75)
        
        w_tum_base = rho[1]*self.gamma + (rho[2] + rho[3])*self.epsilon
        self.warray[2] = w_tum_base*(1-2*s)
        self.warray[3] = w_tum_base - self.r
#        print(self.warray)
        return self.warray
    
    def NextStep(self,rho, B, s):
        # rho: 4x1 array that has [RhoB, RhoC, RhoS, RhoR]
        # B: single number for bone
        # s: current strength of therapy
        # function returns next time step's (rho, B)
        
        self.warray = self.CompFit(rho, B, s)

        #Update bone
        nextB = B + (1/self.B_timescale)*(rho[0]*(2 - B) - rho[1]*B)
        
        #Update Rho
        nextRho = np.zeros(4)
        rho_tot = np.sum(rho)
        
        for i in range(4):
            nextRho[i] = (1 - self.alpha)*rho[i] + self.selection_strength*self.warray[i]*rho[i]*(1 - rho_tot)
        
        return nextRho, nextB

    def runSim(self):
        #runs simulation and creates arrays rho_all: complete list of rhos and B_all: complete list of bone values
        
        self.rho_all[0,:] = self.rhoStart
        # print self.rho_all
        self.B_all[0] = self.BStart
        
        
        for t in range(self.tmax - 1):
            self.fwarray[t+1,:] = self.warray
            self.rho_all[t + 1,:], self.B_all[t + 1] = self.NextStep(self.rho_all[t,:], self.B_all[t], self.s_all[t])
            # print self.rho_all[t+1,0], t 
            if  self.rho_all[t+1, 0] < 0.0001 and t > 1500:    
                self.threshold = 1
                # print "removed" , t
        return self.rho_all, self.B_all,
    
    def runSimAT(self, threshold, time, efficacy, t_begin):
        #runs simulation and creates arrays rho_all: complete list of rhos and B_all: complete list of bone values
        
        self.rho_all[0,:] = self.rhoStart
        
        self.B_all[0] = self.BStart
        
        
        for t in range(self.tmax - 1):
            self.s_all[t] = 0
            if t > t_begin and t%time == 0:
                if(self.rho_all[t,2] + self.rho_all[t,3]) > threshold:
                    self.s_all[t:self.tmax] = min(1,efficacy*1.5)
                else:
                    self.s_all[t:self.tmax] = efficacy*0.5
        
        
            self.rho_all[t + 1,:], self.B_all[t + 1] = self.NextStep(self.rho_all[t,:], self.B_all[t], self.s_all[t])
        
        
            return self.s_all

    def GetMeanSize(self,start=0,end=-1):
        if end < start:
            end = self.tmax
        
        return np.mean(self.rho_all[start:1000,:] , axis=0)

    def GetMeanBone(self,start=0,end=-1):
        if end < start:
            end = self.tmax

        return  np.mean(self.B_all[start:end], axis=0)

    def NormalizeMean(self, start = 0, end =  -1):
        meanArray = np.zeros(4)
        if end < start:
            end = self.tmax
        print end
        meanArray = np.mean(self.rho_all[start:end,:] , axis=0)
        for i in range (4):
            meanArray[i] = meanArray[i]/self.tmax

        return meanArray
    
    
    
    def ChemoPlotShade(self):
        for (block_start, block_end) in self.chemo_blocks:
            plt.axvspan(block_start,block_end,alpha=self.s_all[block_start],color='grey')
    
    def PlotFitness(self):
        plt.plot(self.fwarray,'k-')
        plt.ylim([-1,1])
        plt.show()


    def DoublePlot(self, log_scale = False):
        plt.subplot(211)
        plt.plot(self.B_all, label = "Bone", linewidth = 2)
        plt.ylabel("Relative Amount", fontsize =14)
        # plt.axis('off')
        # plt.tick_params(axis='both', bottom='off', labelbottom='off')
        plt.legend()
        plt.ylim([0,2])
        
        plt.plot([0,self.tmax],[1,1],'k--')
        self.ChemoPlotShade()
        
        plt.subplot(212)
        plt.plot(self.rho_all[:,0], label = "OB", linewidth = 2) 
        plt.plot(self.rho_all[:,1], label = "OC", linewidth = 2) 
        plt.plot(self.rho_all[:,2], label = "T", linewidth = 2) 
        plt.plot(self.rho_all[:,3], label =  "TR", linewidth = 2) 

        plt.legend()
        plt.ylabel("Proportion" , fontsize = 14)
        plt.xlabel('Time',fontsize=14)

        if log_scale:
            plt.yscale('log')
        else:
            plt.ylim([0,1])
        self.ChemoPlotShade()
        
        plt.show()

    def StromaPlot(self, log_scale = False):
        plt.plot(self.rho_all[:,0],'m-', label = "OB",linewidth = 2)
        plt.plot(self.rho_all[:,1],'c--', label = "OC",linewidth = 2)
        plt.plot((self.rho_all[:,0] + self.rho_all[:,1]),'y--', label = "Stroma", linewidth= 2)
        plt.xlabel('Time',fontsize=14)
        plt.legend()
        if log_scale:
            plt.yscale('log')
        else:
            plt.ylim([0,1])
        plt.title('Stroma', fontsize = 20)
        self.ChemoPlotShade()
        
        plt.show()
    
    
    def TumourPlot(self, log_scale = False):
        plt.plot(self.rho_all[:,2] + self.rho_all[:,3],'k-')
        plt.plot(self.rho_all[:,2]/(self.rho_all[:,2] + self.rho_all[:,3]),'r--')
        if log_scale:
            plt.yscale('log')
        else:
            plt.ylim([0,1])
        plt.title('Tumour')
        self.ChemoPlotShade()
        
        plt.show()




F = BoneModel()
F.AllPlayers()
F.AddChemoBlock(200,350, 0.5)
F.runSim()
F.DoublePlot()
F.StromaPlot(log_scale = True)


B = BoneModel()
B.AllPlayers()
B.AddChemoBlock(200, 440, 0.7)
B.AddChemoBlock(880, 920, 0.7)
B.runSim()
print B.GetMeanSize()
B.DoublePlot()
B.StromaPlot(log_scale = True)

A = BoneModel()
A.AllPlayers()
A.ClearChemo()
    # def MakeChemoScheme(self, t_start, t_length, t_break, n_blocks, s_strength):
A.MakeChemoScheme(200,50,100,11,0.7)
A.runSim()
A.DoublePlot()
A.StromaPlot(log_scale = True)

# C = BoneModel()
# C.AllPlayers()
# C.ClearChemo()
# C.AddChemoBlock(200, 400, 0.7)
# C.AddChemoBlock(500,550, 0.7)
# C.AddChemoBlock(650, 700,0.7)
# C.AddChemoBlock(800,850, 0.7)
# C.AddChemoBlock(900,950, 0.7)
# C.AddChemoBlock(1050,1100, 0.7)
# C.runSim()
# print C.GetMeanSize()

# C.DoublePlot()
# C.StromaPlot(log_scale = True) 

# B = BoneModel()
# B.ClearChemo()
# B.AllPlayers()
# for j in range (0, 12):
#     num = x[j:j+1]
#     if(int(num) == 1):
#         B.AddChemoBlock(start + 20*(j+1) , start + 20*(j+2), str)

# B.runSim()


# n = 12 #MAKE SURE TO ALSO change the value in front of b to match n
# val = np.zeros((2**n,7))
# str  = 0.7
# start = 200

# for i in range (0, 2**n):
#     x =  '{0:012b}'.format(i)
#     val[i,0] = int(x)
#     print x 
#     # count = 0   
#     # for c in range (0,n):
#     #     print x,  x[c], c, i 
#     #     if x[c : c+1] == "1":

#     #         count = count + 1

#     #     print count 


#     #Need to re-instantiate B and clear it's memory.
#     B = BoneModel()
#     B.ClearChemo()
#     B.AllPlayers()
#     for j in range (0, n):
#         num = x[j:j+1]
#         if(int(num) == 1):
#             B.AddChemoBlock(start + 40*(j+1) , start + 40*(j+2), str)

#     B.runSim()


#     meansize = B.GetMeanSize()

#     val[i,1] = meansize[0]
#     val[i,2] = meansize[1]
#     val[i,3] = meansize[2]
#     val[i,4] = meansize[3]
#     val[i,5] = meansize[2]+ meansize[3]
# #    print(B.GetMeanBone())
#     val[i,6] = B.GetMeanBone()

#     if B.threshold == 1: 
#         val[i,:] = 0 

#     # if count > 4:
#     #     print "hello"
#     #     val[i,:]= 0

# #    print val
# np.savetxt('egt.csv', val, delimiter = "," )

