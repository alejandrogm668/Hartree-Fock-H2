###########################################################################
# HARTREE FOCK CODE FOR PEDAGOGICAL PURPOSES                              # 
# BASED ON: https://github.com/nickelandcopper/HartreeFockPythonProgram   #
# NEW: ADDED THEORY TO THE NOTEBOOKS                                      #
###########################################################################

import numpy as np 
from scipy.special import erf

#####################################################################
#                           PART1                                   #
#####################################################################

class primitive_gaussian():
    
    def __init__(self,α,coeff,coords):
        self.α = α
        self.coeff = coeff
        self.coords = np.array(coords)
        self.A = (2.0*α/np.pi)**(3/4)      
        
        
def overlap(molecule):
    
    n_basis = len(molecule) # Number of atoms 
    S = np.zeros((n_basis,n_basis)) # Overlap between the atomic orbitals
    
    # The first loop runs over all orbitals
    for i in range(n_basis):
        for j in range(n_basis):
            
            n_primitives_i = len(molecule[i]) # Number of primitives in which orbital i is decomposed
            n_primitives_j = len(molecule[j]) # Number of primitives in which orbital j is decomposed
            
            # The second loop runs over the primitives of the orbitals to be overlap
            for k in range(n_primitives_i): 
                for l in range(n_primitives_j):
                    
                    N = molecule[i][k].A * molecule[j][l].A # Product of primitive normalization constants
                    p = molecule[i][k].α + molecule[j][l].α # α_μ + α_ν
                    q = molecule[i][k].α * molecule[j][l].α / p # α_μ * α_ν / α_μ + α_ν
                    Q = molecule[i][k].coords - molecule[j][l].coords # r_μ - r_ν 
                    Q2 = np.dot(Q,Q) # (r_μ - r_ν)^2
                    
                    S[i,j] += N * molecule[i][k].coeff * molecule[j][l].coeff * np.exp(-q*Q2) * (np.pi/p)**(3/2) # Overlap matrix element
                    
    return S              


#####################################################################
#                           PART2                                   #
#####################################################################

def kinetic(molecule):
            
    n_basis = len(molecule) # Number of atoms 
    T = np.zeros((n_basis,n_basis)) # Overlap between the atomic orbitals
            
    # The first loop runs over all orbitals
    for i in range(n_basis):
        for j in range(n_basis):
            
            n_primitives_i = len(molecule[i]) # Number of primitives in which orbital i is decomposed
            n_primitives_j = len(molecule[j]) # Number of primitives in which orbital j is decomposed
            
            # The second loop runs over the primitives of the orbitals to be overlap
            for k in range(n_primitives_i): 
                for l in range(n_primitives_j):
                    
                    N = molecule[i][k].A * molecule[j][l].A # Product of primitive normalization constants
                    p = molecule[i][k].α + molecule[j][l].α # α_μ + α_ν
                    q = molecule[i][k].α * molecule[j][l].α / p # α_μ * α_ν / α_μ + α_ν
                    Q = molecule[i][k].coords - molecule[j][l].coords # r_μ - r_ν 
                    Q2 = np.dot(Q,Q) # (r_μ - r_ν)^2
                    
                    a = (molecule[i][k].α/p)**2 # (α_μ / α_μ + α_ν)^2
                    
                        
                    
                    S = N * molecule[i][k].coeff * molecule[j][l].coeff * np.exp(-q*Q2) * (np.pi/p)**(3/2) # Overlap matrix element
                    
                    T[i,j] += 3*molecule[j][l].α*S - 2*(molecule[j][l].α)**2 * (1.5/p + a*Q2) * S # Kinetic energy matrix element
                    
    return T 


#####################################################################
#                           PART3                                   #
#####################################################################

def F_0(t):
    if t<1e-6:
        return 1.0 - t/3.0
    else:
        return 0.5*np.sqrt(np.pi/t)*erf(np.sqrt(t))
    
def electron_nuclear_attraction(molecule, atom_coordinates, Z):
    
    n_atoms = len(Z)
    n_basis = len(molecule) # Number of atoms 
    V_en = np.zeros((n_basis,n_basis)) # Overlap between the atomic orbitals
            
    # The first loop runs over all nuclei
    for atom in range(n_atoms):
        # The second loop runs over all orbitals
        for i in range(n_basis):
            for j in range(n_basis):

                n_primitives_i = len(molecule[i]) # Number of primitives in which orbital i is decomposed
                n_primitives_j = len(molecule[j]) # Number of primitives in which orbital j is decomposed

                # The third loop runs over the primitives of the orbitals to be overlap
                for k in range(n_primitives_i): 
                    for l in range(n_primitives_j):

                        N = molecule[i][k].A * molecule[j][l].A # Product of primitive normalization constants
                        p = molecule[i][k].α + molecule[j][l].α # α_μ + α_ν
                        q = molecule[i][k].α * molecule[j][l].α / p # α_μ * α_ν / α_μ + α_ν
                        Q = molecule[i][k].coords - molecule[j][l].coords # r_μ - r_ν 
                        Q2 = np.dot(Q,Q) # (r_μ - r_ν)^2
                        
                        Rp = ( molecule[i][k].α*molecule[i][k].coords + molecule[j][l].α*molecule[j][l].coords)/p
                        RJ = atom_coordinates[atom]
                        R = RJ - Rp
                        R2 = np.dot(R,R)
                                               
                        # Electron-Nucleus potential matrix element    
                        V_en[i,j] -= (2*np.pi/p) * Z[atom] * N * molecule[i][k].coeff * molecule[j][l].coeff * np.exp(-q*Q2) * F_0(p*R2) 
                    
    return V_en



#####################################################################
#                           PART4                                   #
#####################################################################



def electron_electron_repulsion(molecule):
    
    
    n_basis = len(molecule) # Number of atoms 
    V_ee = np.zeros((n_basis,n_basis,n_basis,n_basis)) # Overlap between the atomic orbitals
            
    # This set of loops runs over 4 orbitals because of the interaction matrix    
    for i in range(n_basis):
        for j in range(n_basis):
            for k in range(n_basis):
                for l in range(n_basis):
                    
                    nprimitives_i = len(molecule[i])
                    nprimitives_j = len(molecule[j])
                    nprimitives_k = len(molecule[k])
                    nprimitives_l = len(molecule[l])
                    
                    # This set of loops runs over the primitives in which each orbital is decomposed
                    for ii in range(nprimitives_i):
                        for jj in range(nprimitives_j):
                            for kk in range(nprimitives_k):
                                for ll in range(nprimitives_l):
                                    
                                    N = molecule[i][ii].A * molecule[j][jj].A * molecule[k][kk].A * molecule[l][ll].A 
                                    cicjckcl = molecule[i][ii].coeff * molecule[j][jj].coeff * molecule[k][kk].coeff * molecule[l][ll].coeff
                                    
                                    α_μ, α_ν, α_ρ, α_σ =  molecule[i][ii].α, molecule[j][jj].α, molecule[k][kk].α, molecule[l][ll].α
                                    
                                    α_p = α_μ + α_ν
                                    α_q = α_ρ + α_σ
                                    α_r = α_p * α_q / (α_p + α_q) 
                                    
                                    r_μ, r_ν, r_ρ, r_σ = molecule[i][ii].coords, molecule[j][jj].coords, molecule[k][kk].coords, molecule[l][ll].coords
                                    r_p = (α_μ * r_μ + α_ν * r_ν)/α_p
                                    r_q = (α_ρ * r_ρ + α_σ * r_σ)/α_q
                                    
                                    
                                    R = r_p-r_q
                                    R1 = r_μ-r_ν
                                    R2 = r_ρ-r_σ
                                    
                                    Rsq = np.dot(R,R)
                                    R1sq = np.dot(R1,R1)
                                    R2sq = np.dot(R2,R2)
                                    
                                    term_p = np.exp(-(α_μ*α_ν/α_p)*R1sq)
                                    term_q = np.exp(-(α_ρ*α_σ/α_q)*R2sq)
                                    
                                    V_ee[i,j,k,l] += N * cicjckcl * term_p * term_q * (2*np.pi**(5/2) / np.sqrt((α_p+α_q)))/(α_p*α_q) * F_0(α_r * Rsq)                                    
                                    
    return V_ee



#####################################################################
#                           PART4                                   #
#####################################################################




def nuclear_nuclear_repulsion(atom_coords, Z): 
    
    n_atoms = len(Z) # Number of atoms
    E_NN = 0 
    
    for i in range(n_atoms):
        for j in range(n_atoms):
            if j>i:
                r_ij_vec = atom_coords[i]-atom_coords[j] # Nuclear distance vector
                r_ij = np.sqrt(np.dot(r_ij_vec,r_ij_vec)) # Nuclear distance
                E_NN += Z[i]*Z[j]/r_ij # Repulsion energy
                
    return E_NN     