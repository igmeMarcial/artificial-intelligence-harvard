# IterativePreorderTraversal(root):
#     si root es null:
#         retorna

#     crear una pila vacía
#     empujar root a la pila

#     mientras la pila no esté vacía:
#         nodo = desapilar la pila
#         visitar nodo (imprimir su valor, por ejemplo)

#         si nodo tiene un hijo derecho:
#             empujar hijo derecho a la pila


#         si nodo tiene un hijo izquierdo:
#             empujar hijo izquierdo a la pila
#       1
#      / \
#     2   3
#    / \
#   4   5
# Definición del nodo del árbol binario
class Nodo:

    def __init__(self, valor):
        self.valor = valor
        self.izquierdo = None
        self.derecho = None


# Función para el recorrido en preorder de manera iterativa
def preorder_iterativo(raiz):
    if raiz is None:
        return

    # Crear una pila y añadir el nodo raíz
    pila = []
    pila.append(raiz)

    # Mientras la pila no esté vacía
    while pila:
        # Sacar el nodo de la parte superior de la pila
        nodo_actual = pila.pop()

        # Visitar el nodo (imprimir su valor)
        print(nodo_actual.valor, end=" ")

        # Apilar el hijo derecho primero, ya que el izquierdo debe procesarse antes
        if nodo_actual.derecho:
            pila.append(nodo_actual.derecho)

        # Apilar el hijo izquierdo después
        if nodo_actual.izquierdo:
            pila.append(nodo_actual.izquierdo)


# Ejemplo de uso
if __name__ == "__main__":
    # Crear el árbol del ejemplo:
    #       1
    #      / \
    #     2   3
    #    / \
    #   4   5
    raiz = Nodo(1)
    raiz.izquierdo = Nodo(2)
    raiz.derecho = Nodo(3)
    raiz.izquierdo.izquierdo = Nodo(4)
    raiz.izquierdo.derecho = Nodo(5)

    print("Recorrido Preorder (Iterativo):")
    preorder_iterativo(raiz)
