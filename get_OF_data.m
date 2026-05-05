function [pos,spk,sess,of_sel]=get_OF_data(degu,sess,tet,cl)

    fpath='/mnt/NAS/Degus/merged_files/';



            merge_file=[fpath 'S2020_Mark' degu '-OF-' sess '_merged.db']; 
            sorting_file=[fpath 'S2020_Mark' degu '-OF-' sess '.db_clnew'];
%             try
                load(merge_file,['sess'],'-mat');  % sess es un día (que tiene muchas sesiones)
%             catch err
%                 continue;
%             end
            
%             vs.fpath=[fpath degu '-' sess];
%             vs.mysess=0;
%             vs=get_all_data_nlx(vs,data{i,3});
            load(sorting_file,'-mat');

            
            % select Open fields in each session
            of_sel=find(cellfun(@(x) x(1)=='c' | x(1)=='C' | x(1)=='N', sess.names));
%             of_sel=find(cellfun(@(x) x(1)=='L', sess.names)); % Sesiones cA 
            
            num_of=numel(of_sel);   % n° de sesiones cA 
%  disp(num_of)
            
%             tet=data{i,4}; % tetrode             
%             cl=data{i,5}; %neuron
            
            
            %spike timestamps
            spk=cell(1,num_of);             
%             for k=1:num_of 
            k=numel(of_sel);
                spike_session=['spk_ts_' num2str(of_sel(k)) '_' num2str(tet)];
                db=load(merge_file,spike_session,'-mat');
%                 if isempty(all_clust{of_sel(k),tet})
%                     continue;
%                 end
%                 try 
                    spk{k}=db.(spike_session)(all_clust{of_sel(k),tet}{cl});
%                 catch err
%                     disp(err)
%                     continue;
%                 end
%             end
            
            %degu positions
            pos=cell(1,num_of);
%             for k=1%:num_of
                pos_session=['pos_' num2str(of_sel(k))];
                db=load(merge_file,pos_session,'-mat');
                pos{k}=db.(pos_session);
%             end
            
            
%             
%             [pos{k}.x,pos{k}.y]=clear_straight_lines(pos{k}.x,pos{k}.y,5);
%             
%             [pos{k}.x,pos{k}.y]=clear_switches(pos{k}.x,pos{k}.y,10);
            
            pos=pos{k};
            spk=spk{k};
            of_sel=of_sel(k);
