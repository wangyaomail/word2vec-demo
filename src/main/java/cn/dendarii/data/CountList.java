package cn.dendarii.data;

import java.util.ArrayList;
import java.util.HashMap;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.ToString;

@Data
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = false)
public class CountList<T> extends ArrayList<T> {
    private static final long serialVersionUID = 2868777098497899045L;
    @ToString.Exclude
    public HashMap<T, Integer> countMap;
    public CountList() {
        countMap = new HashMap<T, Integer>();
    }
    public CountList(HashMap<T, Integer> countMap) {
        this.countMap = countMap;
    }
    @Override
    public boolean add(T e) {
        Integer countInteger = countMap.get(e);
        if (countInteger != null) {
            countMap.put(e, countInteger + 1);
        } else {
            countMap.put(e, 1);
        }
        return super.add(e);
    }
}
