package cn.dendarii.data;

import lombok.Data;

@Data
public class Pair<K, V> {
    public K k;
    public V v;
    public Pair(K k,
                V v) {
        this.k = k;
        this.v = v;
    }
}
